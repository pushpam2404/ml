"""Blocking / candidate generation.

For each Source-1 record, find plausibly-matching Source-2/Source-3 records by
inverted-token-index overlap, scoped to the same country (verified in training
data: 0/7.6M ground-truth matches cross a country boundary). Fully vectorized
with pandas explode/merge/groupby -- no per-row python loops -- since sources
run 5-10M rows.

Output shape: DataFrame[source1_entity_id, cand_id, source, overlap] -- one row
per (S1, candidate) pair, already capped to top-K per S1 per source. This is
exactly the candidate set later written to candidate_pairs.tsv and fed to the
feature/model stage.
"""
import os
import sys

import numpy as np
import pandas as pd
import jellyfish

from normalize import addr_tokens, name_tokens

# Per-pass candidate depth. Measured on a 1% sample at 80/30/30 (217 candidates per
# entity): 68% of entities already had 100% blocking recall, so the deep tail bought
# almost nothing while quadrupling the pair count -- which costs both runtime and
# marks on candidate_pairs.tsv, where a smaller set scores better.
GENERAL_TOP_K = 40
ADDR_TOP_K = 20
PIN_TOP_K = 20
GRAM_TOP_K = 15
GRAM_N = 4
# Much tighter than the word indexes. A 4-gram is far less selective, so the join
# fans out as (grams per name) x (df per gram) -- ~18x the average df. At df 2000
# that intermediate runs to hundreds of millions of rows per chunk, which is the
# real cost of this pass, not the index size. Grams that common carry little
# signal anyway and IDF already down-weights them.
GRAM_MAX_DF = 500
S1_CHUNK_ROWS = 40_000  # S1 rows per join batch; bounds the merge intermediate
# Chunks are independent (the demo asserts chunking cannot change the candidate set),
# so they run in parallel. Only via fork: the candidate index is several GB and fork
# shares it copy-on-write, where spawn would pickle a fresh copy into all 7 workers at
# once. macOS fork is unsafe with threads, so it stays serial there.
BLOCK_WORKERS = max(1, (os.cpu_count() or 2) - 1)
BLOCK_PARALLEL = sys.platform != "darwin" and BLOCK_WORKERS > 1
# Parallel chunks must be SMALLER, not the same size. Each worker's merge
# intermediate is private (only the index is shared by fork), so peak memory is
# workers x chunk. Reusing the serial 40k here OOM-killed a worker and broke the
# pool. Sized so workers x this is about one serial chunk's worth in flight.
PARALLEL_CHUNK_ROWS = 12_000
_SHARED = {}
MIN_PAIR_SCORE = 3.0   # minimum summed-IDF for a pair to be worth ranking (0 disables).
MAX_TOKEN_DF = 5000    # drop tokens shared by more than this many candidates in a country.


def _token_index(df: pd.DataFrame, id_col: str, text_col: str, tokenizer) -> pd.DataFrame:
    """Tokenize a normalized text column into long (country, token, <id_col>) rows.

    Tokens are derived here rather than stored on the source frames -- the list
    column costs ~1GB per source if kept, and is only ever needed transiently.
    The tokenizers return each token once per record, so no global de-duplication
    pass over the (much larger) exploded index is needed.
    """
    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        id_col: df[id_col].to_numpy(),
        "token": tokenizer(df[text_col]).to_numpy(),
    }).explode("token")
    return long.dropna(subset=["token"])


def phonetic_tokens(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """Generate Metaphone phonetic codes for name tokens."""
    def codes(text: str):
        # Generate metaphone for each token > 3 chars
        return list(dict.fromkeys(
            jellyfish.metaphone(t) for t in text.split() 
            if len(t) >= 3 and jellyfish.metaphone(t)
        ))
    
    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        id_col: df[id_col].to_numpy(),
        "token": df[text_col].map(codes).to_numpy(),
    }).explode("token")
    return long.dropna(subset=["token"])


def ngram_tokens(df: pd.DataFrame, id_col: str, text_col: str, n: int = GRAM_N) -> pd.DataFrame:
    """Character n-grams of the space-stripped name; catches typos and glued names.

    Whole-word tokens miss a one-character typo entirely ("organic"/"organik" share no
    word) and miss glued names ("wilfordhancock.com"/"wilford hancock"). N-grams of both
    still overlap. Note they do NOT rescue a heavily-mangled string -- "organic" and
    "ornanac" share zero 4-grams -- so this is a typo/concatenation fix, not a
    transliteration fix.
    """
    def grams(text: str):
        s = text.replace(" ", "")
        if len(s) < n:
            return [s] if s else []
        return list(dict.fromkeys(s[i:i + n] for i in range(len(s) - n + 1)))

    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        id_col: df[id_col].to_numpy(),
        "token": df[text_col].map(grams).to_numpy(),
    }).explode("token")
    return long.dropna(subset=["token"])


def _weighted_index(index: pd.DataFrame, n_by_country: pd.Series,
                    max_df: int = MAX_TOKEN_DF) -> pd.DataFrame:
    """Prune over-common (country, token) keys and attach an IDF weight to the rest.

    Two jobs from one doc-frequency pass:
    * pruning -- a token indexing more than max_df candidates would fan the join
      out intractably (and carries almost no signal anyway),
    * weighting -- a shared rare token ("vendome") is far stronger evidence than a
      shared common one ("global"), so candidates are ranked by summed IDF rather
      than by a raw count of shared tokens that treats both alike.

    Each field (name / address / PIN) is weighted independently, so a token that is
    common in addresses but rare in names gets the right weight in each.
    """
    doc_freq = index.groupby(["country", "token"], sort=False)["cand_code"].transform("size").to_numpy()
    keep = doc_freq <= max_df
    index = index.loc[keep].copy()
    n_docs = index["country"].map(n_by_country).to_numpy(dtype="float64")
    index["idf"] = np.log1p(n_docs / doc_freq[keep])
    return index


def _candidates_from_index(s1_long: pd.DataFrame, cand_index: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Join S1 tokens to the candidate index, sum IDF per pair, keep top_k per entity.

    Both id columns are integer codes (`s1_code`, `cand_code`), never strings: the
    join fans out to tens of millions of rows per chunk, and the group-by and sort
    over that dominate blocking runtime. On int32 they are several times faster than
    on object-dtype strings, and the index itself is an order of magnitude smaller.
    Codes are assigned in lexicographic id order, so ordering by code is identical to
    ordering by id and results are unchanged.
    """
    merged = s1_long.merge(cand_index, on=["country", "token"], how="inner")
    if merged.empty:
        return merged.reindex(columns=["s1_code", "cand_code", "overlap"])
    overlap = (
        merged.groupby(["s1_code", "cand_code"], sort=False)["idf"]
        .sum()
        .rename("overlap")
        .reset_index()
    )
    # Drop pairs too weak to reach top-K before sorting. The join yields ~4000
    # candidates per entity of which 30 are kept, and the bulk are pairs sharing a
    # single common (low-IDF) token. Filtering on the SCORE rather than on a count of
    # shared tokens is what makes this safe: a pair sharing one *rare* token scores
    # high and survives, while one sharing a single common token does not.
    if MIN_PAIR_SCORE > 0:
        overlap = overlap.loc[overlap["overlap"].to_numpy() >= MIN_PAIR_SCORE]
        if overlap.empty:
            return overlap.reindex(columns=["s1_code", "cand_code", "overlap"])
    # Deterministic tie-break: scores tie often at the top-K boundary, and without a
    # total order the survivors would depend on row order -- results would vary with
    # chunk size and the run would not be reproducible.
    overlap = overlap.sort_values(
        ["overlap", "cand_code"], ascending=[False, True], kind="stable"
    )
    return overlap.groupby("s1_code", sort=False).head(top_k)


def _s1_long(s1: pd.DataFrame, fields=("name", "addr", "pin")) -> pd.DataFrame:
    """Build the long (country, token, source1_entity_id) frame for the S1 side."""
    parts = []
    if "name" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "name_norm", name_tokens))
    if "phon" in fields:
        parts.append(phonetic_tokens(s1, "source1_entity_id", "name_norm"))
    if "gram" in fields:
        parts.append(ngram_tokens(s1, "source1_entity_id", "name_norm"))
    if "addr" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "addr_norm", addr_tokens))
    if "pin" in fields:
        parts.append(
            s1.loc[s1["pin"] != "", ["country", "pin", "source1_entity_id"]].rename(
                columns={"pin": "token"}
            )
        )
    return pd.concat(parts, ignore_index=True)


def _block_one_chunk(bounds):
    """One S1 row-range against the inherited index. Body mirrors the serial loop."""
    start, stop = bounds
    chunk = _SHARED["s1"].iloc[start:stop]
    s1_long = _s1_long(chunk, _SHARED["fields"])
    codes, uniques = pd.factorize(s1_long["source1_entity_id"])
    s1_long = s1_long.drop(columns=["source1_entity_id"])
    s1_long["s1_code"] = codes.astype("int32")
    res = _candidates_from_index(s1_long, _SHARED["index"], _SHARED["top_k"])
    return res.assign(source1_entity_id=uniques.take(res["s1_code"].to_numpy())) \
              .drop(columns=["s1_code"])


def _block_chunked(s1: pd.DataFrame, index: pd.DataFrame, top_k: int, fields,
                   chunk_rows: int = S1_CHUNK_ROWS, label: str = "") -> pd.DataFrame:
    """Run the token join in S1 row-chunks, returning [source1_entity_id, cand_code, overlap].

    The join's intermediate scales with the number of S1 rows fed in, so a
    single-shot merge over millions of S1 rows would materialise billions of rows
    before top-K ever trims it. Chunking bounds peak memory: the candidate index
    is built once by the caller, and only the already-capped per-chunk results are
    accumulated.

    S1 ids are coded to ints per chunk (cheap -- one pass over the chunk's own tokens)
    so the fanned-out group-by and sort run on integers, then mapped back on the small
    capped result.
    """
    par = BLOCK_PARALLEL and len(s1) > PARALLEL_CHUNK_ROWS
    step = min(chunk_rows, PARALLEL_CHUNK_ROWS) if par else chunk_rows
    bounds = [(st, min(st + step, len(s1))) for st in range(0, len(s1), step)]
    n_chunks = max(1, len(bounds))
    if par and len(bounds) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # Populate the globals BEFORE forking: children inherit them copy-on-write, so
        # the multi-GB index is never copied or pickled.
        _SHARED["s1"], _SHARED["index"] = s1, index
        _SHARED["top_k"], _SHARED["fields"] = top_k, fields
        try:
            with ProcessPoolExecutor(max_workers=BLOCK_WORKERS,
                                     mp_context=mp.get_context("fork")) as ex:
                out = list(ex.map(_block_one_chunk, bounds))
            print(f"    [block{label}] {n_chunks} chunks on {BLOCK_WORKERS} workers", flush=True)
            _SHARED.clear()
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
                columns=["source1_entity_id", "cand_code", "overlap"])
        except Exception as exc:
            print(f"    [block{label}] parallel failed ({exc!r})", flush=True)
            # A broken pool means a worker was OOM-killed, so halve and retry while
            # still parallel. Dropping straight to serial costs ~7x, which on the
            # full test set is the difference between hours and missing the deadline.
            for retry_step in (step // 2, step // 4):
                if retry_step < 500:
                    break
                rb = [(st, min(st + retry_step, len(s1)))
                      for st in range(0, len(s1), retry_step)]
                try:
                    with ProcessPoolExecutor(max_workers=max(2, BLOCK_WORKERS // 2),
                                             mp_context=mp.get_context("fork")) as ex:
                        out = list(ex.map(_block_one_chunk, rb))
                    print(f"    [block{label}] recovered at step={retry_step}", flush=True)
                    _SHARED.clear()
                    return pd.concat(out, ignore_index=True)
                except Exception as exc2:
                    print(f"    [block{label}] step={retry_step} failed ({exc2!r})", flush=True)
            _SHARED.clear()
            print(f"    [block{label}] serial fallback", flush=True)

    out = []
    for i, start in enumerate(range(0, len(s1), chunk_rows), 1):
        chunk = s1.iloc[start:start + chunk_rows]
        s1_long = _s1_long(chunk, fields)
        codes, uniques = pd.factorize(s1_long["source1_entity_id"])
        s1_long = s1_long.drop(columns=["source1_entity_id"])
        s1_long["s1_code"] = codes.astype("int32")
        res = _candidates_from_index(s1_long, index, top_k)
        res = res.assign(source1_entity_id=uniques.take(res["s1_code"].to_numpy())) \
                 .drop(columns=["s1_code"])
        out.append(res)
        if n_chunks > 1:
            print(f"    [block{label}] chunk {i}/{n_chunks}", flush=True)
    if not out:
        return pd.DataFrame(columns=["source1_entity_id", "cand_code", "overlap"])
    return pd.concat(out, ignore_index=True)


def block_one_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str,
                     chunk_rows: int = S1_CHUNK_ROWS) -> pd.DataFrame:
    """Return top candidates per S1 from a single candidate source (S2 or S3) via multi-pass."""
    n_by_country = cand.groupby("country", sort=False).size()
    # Code candidate ids to int32 once, in lexicographic order, so every downstream
    # group-by/sort works on integers and ordering by code == ordering by id.
    cand_codes, cand_uniques = pd.factorize(cand["entity_id"], sort=True)
    coded = cand[["country", "name_norm", "addr_norm", "pin"]].copy()
    coded["cand_code"] = cand_codes.astype("int32")

    # 1. Build individual indices
    name_idx = _weighted_index(_token_index(coded, "cand_code", "name_norm", name_tokens), n_by_country)
    addr_idx = _weighted_index(_token_index(coded, "cand_code", "addr_norm", addr_tokens), n_by_country)
    raw_pin_idx = coded.loc[coded["pin"] != "", ["country", "pin", "cand_code"]].rename(
        columns={"pin": "token"}
    )
    pin_idx = _weighted_index(raw_pin_idx, n_by_country)
    phonetic_idx = _weighted_index(phonetic_tokens(coded, "cand_code", "name_norm"), n_by_country)
    # Tighter max_df than the word indexes: a 4-gram is far less selective than a word,
    # so the common ones would fan the join out without adding evidence.
    gram_idx = _weighted_index(ngram_tokens(coded, "cand_code", "name_norm"),
                               n_by_country, max_df=GRAM_MAX_DF)

    results = []

    # PASS 1: General Pass (Name + Address + Phonetic)
    full_index = pd.concat([name_idx, addr_idx, phonetic_idx], ignore_index=True)
    res_general = _block_chunked(
        s1, full_index, GENERAL_TOP_K, ("name", "addr", "phon"), chunk_rows,
        f" {source_label}-general"
    )
    results.append(res_general)
    del full_index

    # PASS 2: Address-heavy Pass (For severe name transliteration mismatch)
    res_addr = _block_chunked(
        s1, addr_idx, ADDR_TOP_K, ("addr",), chunk_rows, f" {source_label}-addr"
    )
    results.append(res_addr)

    # PASS 3: PIN-heavy Pass (For shared PIN + slight name/address overlap)
    # We mix pin and name so we don't just return everyone in the same pin code blindly,
    # but give high weight to pin.
    pin_name_idx = pd.concat([pin_idx, name_idx], ignore_index=True)
    res_pin = _block_chunked(
        s1, pin_name_idx, PIN_TOP_K, ("pin", "name"), chunk_rows, f" {source_label}-pin"
    )
    results.append(res_pin)

    # PASS 4: character n-grams (typos, glued names). Its own pass on purpose -- ranking
    # is summed IDF, and a name yields ~18 n-grams against ~3 words, so folding grams
    # into the general index would let gram mass outvote every word match.
    res_gram = _block_chunked(
        s1, gram_idx, GRAM_TOP_K, ("gram",), chunk_rows, f" {source_label}-gram"
    )
    results.append(res_gram)
    del pin_name_idx, name_idx, addr_idx, pin_idx, phonetic_idx, raw_pin_idx, gram_idx, coded

    # Union all passes
    result = pd.concat(results, ignore_index=True)

    # Deduplicate: if same S1 and cand_code found in multiple passes, keep the max overlap
    result = result.groupby(["source1_entity_id", "cand_code"], as_index=False)["overlap"].max()

    # Map candidate codes back to real ids
    result["cand_id"] = cand_uniques.take(result["cand_code"].to_numpy())
    result = result.drop(columns=["cand_code"])

    result["source"] = source_label
    return result[["source1_entity_id", "cand_id", "source", "overlap"]]


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                        chunk_rows: int = S1_CHUNK_ROWS) -> pd.DataFrame:
    """Full candidate generation: block against S2 and S3 independently, concat."""
    cand2 = block_one_source(s1, s2, "S2", chunk_rows)
    cand3 = block_one_source(s1, s3, "S3", chunk_rows)
    return pd.concat([cand2, cand3], ignore_index=True)


def demo():
    from normalize import prep_source

    # MIN_PAIR_SCORE is calibrated against the real corpus, where idf = log1p(n_docs/df)
    # is large because n_docs is in the millions. On a toy corpus every score is ~1, so
    # the production threshold would reject everything -- scale-dependence is a known
    # limitation of an absolute cutoff. Exercise the blocking mechanics without it.
    global MIN_PAIR_SCORE
    saved, MIN_PAIR_SCORE = MIN_PAIR_SCORE, 0.0

    s1 = prep_source(pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["Iris Brothers Pvt Ltd", "Lonely Business"],
        "business_address": ["123 Main St, Springfield, IL 62701", "9 Nowhere Rd"],
        "country": ["India", "US"],
    })).rename(columns={"entity_id": "source1_entity_id"})

    s2 = prep_source(pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ["Iris Brothers Private Limited", "Totally Unrelated Co"],
        "business_address": ["123 MAIN STREET, SPRINGFIELD, IL", "1 Elsewhere Ave"],
        "country": ["India", "US"],
    }))
    s3 = prep_source(pd.DataFrame({
        "entity_id": ["S3-1"],
        "business_name": ["Nothing Alike Whatsoever"],
        "business_address": ["999 Faraway Blvd"],
        "country": ["US"],
    }))

    cands = generate_candidates(s1, s2, s3)
    got = set(cands.loc[cands["source1_entity_id"] == "S1-1", "cand_id"])
    assert "S2-1" in got, got
    assert "S2-2" not in got, got

    # chunking must not change which candidates survive
    one_shot = generate_candidates(s1, s2, s3, chunk_rows=10**9)
    chunked = generate_candidates(s1, s2, s3, chunk_rows=1)
    key = ["source1_entity_id", "cand_id"]
    assert (one_shot.sort_values(key)[key].reset_index(drop=True)
            .equals(chunked.sort_values(key)[key].reset_index(drop=True))), \
        "chunking changed the candidate set"

    MIN_PAIR_SCORE = saved
    print("blocking.py demo OK")


if __name__ == "__main__":
    demo()
