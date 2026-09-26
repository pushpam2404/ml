"""Feature engineering for (S1, candidate) pairs that survived blocking.

Input: a pairs dataframe with source1_entity_id/cand_id plus the two sides'
prepped columns (name_norm, addr_norm, pin, house_no, name_toks, addr_toks)
already merged in with suffixes _1/_2. Vectorized with rapidfuzz.process
batch scoring where possible.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

FEATURE_COLS = [
    "name_jaccard", "name_trigram_jaccard", "name_token_sort_ratio",
    "name_partial_ratio", "name_len_diff",
    "addr_jaccard", "addr_ratio", "pin_match", "house_match",
    "addr_empty_either",
]


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _trigrams(s: str) -> set:
    s = s.replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def build_pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """pairs must have *_1 (S1 side) and *_2 (candidate side) columns."""
    out = pd.DataFrame(index=pairs.index)

    out["name_jaccard"] = [
        _jaccard(a, b) for a, b in zip(pairs["name_toks_1"], pairs["name_toks_2"])
    ]
    out["name_trigram_jaccard"] = [
        _jaccard(list(_trigrams(a)), list(_trigrams(b)))
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]
    out["name_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]
    out["name_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]
    out["name_len_diff"] = (
        pairs["name_norm_1"].str.len() - pairs["name_norm_2"].str.len()
    ).abs()

    out["addr_jaccard"] = [
        _jaccard(a, b) for a, b in zip(pairs["addr_toks_1"], pairs["addr_toks_2"])
    ]
    out["addr_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(pairs["addr_norm_1"], pairs["addr_norm_2"])
    ]
    out["pin_match"] = (
        (pairs["pin_1"] != "") & (pairs["pin_1"] == pairs["pin_2"])
    ).astype(float)
    out["house_match"] = (
        (pairs["house_no_1"] != "") & (pairs["house_no_1"] == pairs["house_no_2"])
    ).astype(float)
    out["addr_empty_either"] = (
        (pairs["addr_norm_1"] == "") | (pairs["addr_norm_2"] == "")
    ).astype(float)

    return out[FEATURE_COLS]


def demo():
    pairs = pd.DataFrame({
        "name_toks_1": [["iris", "brothers"], ["acme"]],
        "name_toks_2": [["iris", "brothers"], ["zenith"]],
        "name_norm_1": ["iris brothers pvt ltd", "acme co"],
        "name_norm_2": ["iris brothers pvt ltd", "zenith co"],
        "addr_toks_1": [["main", "springfield"], ["oak"]],
        "addr_toks_2": [["main", "springfield"], ["pine"]],
        "addr_norm_1": ["123 main st springfield", "1 oak rd"],
        "addr_norm_2": ["123 main st springfield", "2 pine rd"],
        "pin_1": ["62701", ""],
        "pin_2": ["62701", ""],
        "house_no_1": ["123", "1"],
        "house_no_2": ["123", "2"],
    })
    feats = build_pair_features(pairs)
    assert feats.loc[0, "name_jaccard"] == 1.0
    assert feats.loc[0, "pin_match"] == 1.0
    assert feats.loc[1, "name_jaccard"] == 0.0
    print("features.py demo OK")


if __name__ == "__main__":
    demo()
