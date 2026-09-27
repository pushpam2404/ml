"""Chunked feature-building and model scoring over candidate pairs.

Both the training and prediction paths go through here, so the features a model
is trained on and the features it scores at inference time cannot drift apart.

Feature building is the slowest stage of the whole pipeline -- ~200M pairs, each
needing half a dozen string-similarity measures that are branchy, variable-length
work no vectorised array op covers. It is pure CPU and embarrassingly parallel per
pair, so it is spread across worker processes; everything else here is unchanged.
"""
import os

import numpy as np
import pandas as pd

from data_io import attach_pair_columns
from features import build_pair_features

PAIR_CHUNK_ROWS = 2_000_000
# Rows handed to one worker. Small enough that pickling the hydrated strings stays
# cheap and slow chunks cannot stall the pool, large enough to amortise the handoff.
WORKER_CHUNK_ROWS = 100_000
# Leave a core for the parent, which is meanwhile hydrating the next chunk.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)
# Below this, process startup and pickling cost more than the work saved.
PARALLEL_MIN_ROWS = 200_000

_POOL = None


def _get_pool():
    """One pool for the whole run; respawning per chunk would cost more than it saves."""
    global _POOL
    if _POOL is None:
        from concurrent.futures import ProcessPoolExecutor
        # Each worker is single-threaded on purpose: N processes each spawning N BLAS
        # threads oversubscribes the CPU and runs slower than one thread apiece.
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        os.environ.update(env)
        _POOL = ProcessPoolExecutor(max_workers=N_WORKERS)
    return _POOL


def _hydrated_features(hydrated: pd.DataFrame) -> pd.DataFrame:
    """Features for an already-hydrated chunk, split across worker processes.

    Order is preserved (``map`` yields in submission order), so the result is
    row-aligned with the input exactly as the serial path would be -- the identity
    score.py's demo asserts.
    """
    if len(hydrated) < PARALLEL_MIN_ROWS or N_WORKERS == 1:
        return build_pair_features(hydrated)

    slices = [
        hydrated.iloc[i:i + WORKER_CHUNK_ROWS]
        for i in range(0, len(hydrated), WORKER_CHUNK_ROWS)
    ]
    try:
        parts = list(_get_pool().map(build_pair_features, slices))
    except Exception as exc:  # pool died (OOM-killed worker, fork issue, ...)
        # Correct output matters more than speed: fall back rather than lose the run.
        global _POOL
        _POOL = None
        print(f"    [features] parallel path failed ({exc!r}); falling back to serial",
              flush=True)
        return build_pair_features(hydrated)
    return pd.concat(parts, ignore_index=True)


def iter_pair_features(pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame,
                       chunk_rows: int = PAIR_CHUNK_ROWS, verbose: bool = True):
    """Yield (chunk_of_pairs, feature_frame) tuples, hydrating one chunk at a time."""
    n_chunks = max(1, -(-len(pairs) // chunk_rows))
    for i, start in enumerate(range(0, len(pairs), chunk_rows), 1):
        chunk = pairs.iloc[start:start + chunk_rows]
        feats = _hydrated_features(attach_pair_columns(chunk, s1_side, cand_side))
        if verbose and n_chunks > 1:
            print(f"    [features] chunk {i}/{n_chunks} ({len(chunk)} pairs)", flush=True)
        yield chunk, feats


def build_features(pairs: pd.DataFrame, s1_side: pd.DataFrame, cand_side: pd.DataFrame,
                   chunk_rows: int = PAIR_CHUNK_ROWS, verbose: bool = True) -> pd.DataFrame:
    """Feature frame for every pair, row-aligned with ``pairs``."""
    parts = [f for _, f in iter_pair_features(pairs, s1_side, cand_side, chunk_rows, verbose)]
    if not parts:
        return _hydrated_features(attach_pair_columns(pairs, s1_side, cand_side))
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
    assert str(one["name_jaccard"].dtype) == "float32", one.dtypes

    # The parallel path must produce byte-identical features to the serial one --
    # a split that silently reorders rows would mislabel every pair downstream.
    global PARALLEL_MIN_ROWS, WORKER_CHUNK_ROWS
    hydrated = attach_pair_columns(pairs, s1_side, cand_side)
    serial = build_pair_features(hydrated)
    saved_min, saved_w = PARALLEL_MIN_ROWS, WORKER_CHUNK_ROWS
    PARALLEL_MIN_ROWS, WORKER_CHUNK_ROWS = 0, 1  # force every row into its own task
    try:
        parallel = _hydrated_features(hydrated)
    finally:
        PARALLEL_MIN_ROWS, WORKER_CHUNK_ROWS = saved_min, saved_w
    assert serial.equals(parallel), "parallel features differ from serial"

    print(f"score.py demo OK (N_WORKERS={N_WORKERS})")


if __name__ == "__main__":
    demo()
