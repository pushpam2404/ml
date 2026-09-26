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
import numpy as np
import pandas as pd

from normalize import addr_tokens, name_tokens

TOP_K = 30
FALLBACK_TOP_K = 15
S1_CHUNK_ROWS = 25_000  # S1 rows per join batch; bounds the merge intermediate
MIN_PAIR_SCORE = 10.0  # minimum summed-IDF for a pair to be worth ranking (0 disables).
                        # Calibrated: among candidates that actually survive into top-K,
                        # the 1st percentile score is 8.9, so pruning at 10 discards the
                        # bulk of the join's output (~4000 candidates/entity are scored to
                        # keep 30) at a measured cost of 5 true pairs in 17,262
                        # (recall 0.6837 -> 0.6834) while slightly shrinking the candidate
                        # set. At 16 recall drops to 0.644, at 22 to 0.486.
MAX_TOKEN_DF = 3000  # drop tokens shared by more than this many candidates in a country.
                      # Calibrated against real data: median token doc-freq is 1, but a
                      # long tail of un-normalized suffix typos (e.g. "limittedd" at
                      # 300k+) needs pruning while still-useful common words ("pizza"
                      # ~5.6k, "global" ~22k) must survive -- a too-low cutoff (200) was
                      # zeroing out ALL tokens for common-word business names entirely.


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
    if "addr" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "addr_norm", addr_tokens))
    if "pin" in fields:
        parts.append(
            s1.loc[s1["pin"] != "", ["country", "pin", "source1_entity_id"]].rename(
                columns={"pin": "token"}
            )
        )
    return pd.concat(parts, ignore_index=True)


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
    out = []
    n_chunks = max(1, -(-len(s1) // chunk_rows))
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
    """Return top-K candidates per S1 from a single candidate source (S2 or S3)."""
    n_by_country = cand.groupby("country", sort=False).size()
    # Code candidate ids to int32 once, in lexicographic order, so every downstream
    # group-by/sort works on integers and ordering by code == ordering by id.
    cand_codes, cand_uniques = pd.factorize(cand["entity_id"], sort=True)
    coded = cand[["country", "name_norm", "addr_norm", "pin"]].copy()
    coded["cand_code"] = cand_codes.astype("int32")

    name_idx = _weighted_index(_token_index(coded, "cand_code", "name_norm", name_tokens), n_by_country)
    addr_idx = _weighted_index(_token_index(coded, "cand_code", "addr_norm", addr_tokens), n_by_country)
    raw_pin_idx = coded.loc[coded["pin"] != "", ["country", "pin", "cand_code"]].rename(
        columns={"pin": "token"}
    )
    pin_idx = _weighted_index(raw_pin_idx, n_by_country)
    full_index = pd.concat([name_idx, addr_idx, pin_idx], ignore_index=True)
    del name_idx, pin_idx, raw_pin_idx, coded, cand_codes

    result = _block_chunked(
        s1, full_index, TOP_K, ("name", "addr", "pin"), chunk_rows, f" {source_label}"
    )
    del full_index

    # Fallback: S1 ids with zero candidates so far get a relaxed address-only pass.
    covered = set(result["source1_entity_id"].unique())
    uncovered_s1 = s1.loc[~s1["source1_entity_id"].isin(covered)]
    if not uncovered_s1.empty:
        relaxed = _block_chunked(
            uncovered_s1, addr_idx, FALLBACK_TOP_K, ("addr",), chunk_rows,
            f" {source_label}-fallback",
        )
        result = pd.concat([result, relaxed], ignore_index=True)

    # map candidate codes back to real ids only on the small capped result
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
    print("blocking.py demo OK")


if __name__ == "__main__":
    demo()
