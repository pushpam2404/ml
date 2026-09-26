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
import pandas as pd

TOP_K = 30
FALLBACK_TOP_K = 5
MAX_TOKEN_DF = 200  # drop tokens shared by more than this many candidates in a country
                     # (e.g. "mumbai", "main") -- otherwise the merge fans out to O(n^2)


def _token_index(df: pd.DataFrame, id_col: str, tok_col: str) -> pd.DataFrame:
    """Explode a token-list column into long (country, token, id) rows."""
    long = df[["country", id_col, tok_col]].explode(tok_col)
    long = long.rename(columns={tok_col: "token", id_col: "cand_id"})
    return long.dropna(subset=["token"])


def _prune_common_tokens(index: pd.DataFrame, max_df: int = MAX_TOKEN_DF) -> pd.DataFrame:
    """Drop (country, token) keys that index more candidates than max_df.

    Overly common tokens (city names, "main", ...) are useless for blocking and
    make the join blow up on 5-10M row sources -- prune them before the merge.
    """
    doc_freq = index.groupby(["country", "token"], sort=False)["cand_id"].transform("size")
    return index.loc[doc_freq <= max_df]


def _candidates_from_index(s1_long: pd.DataFrame, cand_index: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Join S1 tokens to the candidate index on (country, token), tally overlap, keep top_k."""
    merged = s1_long.merge(cand_index, on=["country", "token"], how="inner")
    if merged.empty:
        return merged.assign(overlap=pd.Series(dtype=int))
    overlap = (
        merged.groupby(["source1_entity_id", "cand_id"], sort=False)
        .size()
        .rename("overlap")
        .reset_index()
    )
    overlap = overlap.sort_values("overlap", ascending=False)
    return overlap.groupby("source1_entity_id", sort=False).head(top_k)


def block_one_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Return top-K candidates per S1 from a single candidate source (S2 or S3)."""
    name_idx = _prune_common_tokens(_token_index(cand, "entity_id", "name_toks"))
    addr_idx = _prune_common_tokens(_token_index(cand, "entity_id", "addr_toks"))
    pin_idx = cand.loc[cand["pin"] != "", ["country", "pin", "entity_id"]].rename(
        columns={"pin": "token", "entity_id": "cand_id"}
    )  # PIN codes are naturally low-cardinality; no pruning needed
    full_index = pd.concat([name_idx, addr_idx, pin_idx], ignore_index=True)

    s1_name = _token_index(s1, "source1_entity_id", "name_toks").rename(columns={"cand_id": "source1_entity_id"})
    s1_addr = _token_index(s1, "source1_entity_id", "addr_toks").rename(columns={"cand_id": "source1_entity_id"})
    s1_pin = s1.loc[s1["pin"] != "", ["country", "pin", "source1_entity_id"]].rename(columns={"pin": "token"})
    s1_long = pd.concat([s1_name, s1_addr, s1_pin], ignore_index=True)

    result = _candidates_from_index(s1_long, full_index, TOP_K)

    # Fallback: S1 ids with zero candidates so far get a relaxed city/state-token pass.
    covered = set(result["source1_entity_id"].unique())
    uncovered_s1 = s1.loc[~s1["source1_entity_id"].isin(covered)]
    if not uncovered_s1.empty:
        # relaxed key: any address token (already includes city/state words), no name requirement
        relaxed_s1_long = _token_index(uncovered_s1, "source1_entity_id", "addr_toks").rename(
            columns={"cand_id": "source1_entity_id"}
        )
        relaxed = _candidates_from_index(relaxed_s1_long, addr_idx, FALLBACK_TOP_K)
        result = pd.concat([result, relaxed], ignore_index=True)

    result["source"] = source_label
    return result[["source1_entity_id", "cand_id", "source", "overlap"]]


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Full candidate generation: block against S2 and S3 independently, concat."""
    cand2 = block_one_source(s1, s2, "S2")
    cand3 = block_one_source(s1, s3, "S3")
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
