"""Chunked feature-building and model scoring over candidate pairs.

Both the training and prediction paths go through here, so the features a model
is trained on and the features it scores at inference time cannot drift apart.
"""
import numpy as np
import pandas as pd

from data_io import attach_pair_columns
from features import build_pair_features

PAIR_CHUNK_ROWS = 1_000_000


def iter_pair_features(pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame,
                       chunk_rows: int = PAIR_CHUNK_ROWS, verbose: bool = True):
    """Yield (chunk_of_pairs, feature_frame) tuples, hydrating one chunk at a time."""
    n_chunks = max(1, -(-len(pairs) // chunk_rows))
    for i, start in enumerate(range(0, len(pairs), chunk_rows), 1):
        chunk = pairs.iloc[start:start + chunk_rows]
        feats = build_pair_features(attach_pair_columns(chunk, s1_side, cand_side))
        if verbose and n_chunks > 1:
            print(f"    [features] chunk {i}/{n_chunks} ({len(chunk)} pairs)", flush=True)
        yield chunk, feats


def build_features(pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame,
                   chunk_rows: int = PAIR_CHUNK_ROWS, verbose: bool = True) -> pd.DataFrame:
    """Feature frame for every pair, row-aligned with ``pairs``."""
    parts = [f for _, f in iter_pair_features(pairs, s1_side, cand_side, chunk_rows, verbose)]
    if not parts:
        return build_pair_features(attach_pair_columns(pairs, s1_side, cand_side))
    return pd.concat(parts, ignore_index=True)


def predict_proba(booster, pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame,
                  feature_cols, chunk_rows: int = PAIR_CHUNK_ROWS,
                  verbose: bool = True) -> np.ndarray:
    """Match probability per pair, scored chunk by chunk so memory stays bounded."""
    import xgboost as xgb

    out = []
    for _, feats in iter_pair_features(pairs, s1_side, cand_side, chunk_rows, verbose):
        out.append(booster.predict(xgb.DMatrix(feats[feature_cols])))
    return np.concatenate(out) if out else np.empty(0, dtype="float32")


def demo():
    from normalize import prep_source
    from data_io import build_side_lookups

    s1 = prep_source(pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Iris Brothers Pvt Ltd"],
        "business_address": ["123 Main St, Springfield, IL 62701"],
        "country": ["India"],
    })).rename(columns={"entity_id": "source1_entity_id"})
    s2 = prep_source(pd.DataFrame({
        "entity_id": ["S2-1", "S2-2"],
        "business_name": ["Iris Brothers Private Limited", "Unrelated Co"],
        "business_address": ["123 MAIN STREET, SPRINGFIELD, IL", "9 Elsewhere Ave"],
        "country": ["India", "India"],
    }))
    s3 = prep_source(pd.DataFrame({
        "entity_id": ["S3-1"],
        "business_name": ["Nothing Alike"],
        "business_address": ["999 Faraway Blvd"],
        "country": ["India"],
    }))
    pairs = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-1"],
        "cand_id": ["S2-1", "S2-2", "S3-1"],
    })
    s1_side, cand_side = build_side_lookups(s1, s2, s3)

    # chunked must equal one-shot
    one = build_features(pairs, s1_side, cand_side, chunk_rows=10**9, verbose=False)
    many = build_features(pairs, s1_side, cand_side, chunk_rows=1, verbose=False)
    assert one.equals(many), "chunked features differ from one-shot"
    assert one.loc[0, "name_jaccard"] == 1.0, one.loc[0].to_dict()
    assert one.loc[1, "name_jaccard"] == 0.0
    print("score.py demo OK")


if __name__ == "__main__":
    demo()
