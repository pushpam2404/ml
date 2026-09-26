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
MAX_TOKEN_DF = 3000  # drop tokens shared by more than this many candidates in a country.
                      # Calibrated against real data: median token doc-freq is 1, but a
                      # long tail of un-normalized suffix typos (e.g. "limittedd" at
                      # 300k+) needs pruning while still-useful common words ("pizza"
                      # ~5.6k, "global" ~22k) must survive -- a too-low cutoff (200) was
                      # zeroing out ALL tokens for common-word business names entirely.


def _token_index(df: pd.DataFrame, id_col: str, text_col: str, tokenizer) -> pd.DataFrame:
    """Tokenize a normalized text column into long (country, token, id) rows.

    Tokens are derived here rather than stored on the source frames -- the list
    column costs ~1GB per source if kept, and is only ever needed transiently.
    """
    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        "cand_id": df[id_col].to_numpy(),
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
    index = index.drop_duplicates(["country", "token", "cand_id"]).reset_index(drop=True)
    doc_freq = index.groupby(["country", "token"], sort=False)["cand_id"].transform("size").to_numpy()
    keep = doc_freq <= max_df
    index = index.loc[keep].copy()
    n_docs = index["country"].map(n_by_country).to_numpy(dtype="float64")
    index["idf"] = np.log1p(n_docs / doc_freq[keep])
    return index


def _candidates_from_index(s1_long: pd.DataFrame, cand_index: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Join S1 tokens to the candidate index on (country, token), sum IDF, keep top_k."""
    merged = s1_long.merge(cand_index, on=["country", "token"], how="inner")
    if merged.empty:
        return merged.reindex(columns=["source1_entity_id", "cand_id", "overlap"])
    overlap = (
        merged.groupby(["source1_entity_id", "cand_id"], sort=False)["idf"]
        .sum()
        .rename("overlap")
        .reset_index()
    )
    # Deterministic tie-break on cand_id: scores tie often at the top-K boundary, and
    # without a total order the surviving candidates would depend on row order -- which
    # would make results vary with chunk size and the run non-reproducible.
    overlap = overlap.sort_values(
        ["overlap", "cand_id"], ascending=[False, True], kind="stable"
    )
    return overlap.groupby("source1_entity_id", sort=False).head(top_k)


def _s1_long(s1: pd.DataFrame, fields=("name", "addr", "pin")) -> pd.DataFrame:
    """Build the long (country, token, source1_entity_id) frame for the S1 side."""
    parts = []
    if "name" in fields:
        parts.append(
            _token_index(s1, "source1_entity_id", "name_norm", name_tokens).rename(
                columns={"cand_id": "source1_entity_id"}
            )
        )
    if "addr" in fields:
        parts.append(
            _token_index(s1, "source1_entity_id", "addr_norm", addr_tokens).rename(
                columns={"cand_id": "source1_entity_id"}
            )
        )
    if "pin" in fields:
        parts.append(
            s1.loc[s1["pin"] != "", ["country", "pin", "source1_entity_id"]].rename(
                columns={"pin": "token"}
            )
        )
    return pd.concat(parts, ignore_index=True)


def _block_chunked(s1: pd.DataFrame, index: pd.DataFrame, top_k: int, fields,
                   chunk_rows: int = S1_CHUNK_ROWS, label: str = "") -> pd.DataFrame:
    """Run the token join in S1 row-chunks.

    The join's intermediate scales with the number of S1 rows fed in, so a
    single-shot merge over millions of S1 rows would materialise billions of rows
    before top-K ever trims it. Chunking bounds peak memory: the candidate index
    is built once by the caller, and only the already-capped per-chunk results are
    accumulated.
    """
    out = []
    n_chunks = max(1, -(-len(s1) // chunk_rows))
    for i, start in enumerate(range(0, len(s1), chunk_rows), 1):
        chunk = s1.iloc[start:start + chunk_rows]
        out.append(_candidates_from_index(_s1_long(chunk, fields), index, top_k))
        if n_chunks > 1:
            print(f"    [block{label}] chunk {i}/{n_chunks}", flush=True)
    if not out:
        return pd.DataFrame(columns=["source1_entity_id", "cand_id", "overlap"])
    return pd.concat(out, ignore_index=True)


def block_one_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str,
                     chunk_rows: int = S1_CHUNK_ROWS) -> pd.DataFrame:
    """Return top-K candidates per S1 from a single candidate source (S2 or S3)."""
    n_by_country = cand.groupby("country", sort=False).size()
    name_idx = _weighted_index(_token_index(cand, "entity_id", "name_norm", name_tokens), n_by_country)
    addr_idx = _weighted_index(_token_index(cand, "entity_id", "addr_norm", addr_tokens), n_by_country)
    raw_pin_idx = cand.loc[cand["pin"] != "", ["country", "pin", "entity_id"]].rename(
        columns={"pin": "token", "entity_id": "cand_id"}
    )
    pin_idx = _weighted_index(raw_pin_idx, n_by_country)
    full_index = pd.concat([name_idx, addr_idx, pin_idx], ignore_index=True)
    del name_idx, pin_idx, raw_pin_idx

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
