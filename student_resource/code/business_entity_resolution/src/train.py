"""Train the candidate classifier on the training set.

Usage:
    python3 train.py --data-dir ../../../dataset --model-out ../model \
        [--sample-frac 0.05] [--seed 42]

Pipeline: load train sources -> block candidates -> label with ground truth ->
split by S1 id -> engineer features -> train XGBoost -> pick a macro-F0.5-
optimal probability threshold on the held-out split -> save model+threshold+
feature list to --model-out/model.json (booster) and model_meta.json.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from blocking import generate_candidates
from data_io import load_source, build_side_lookups
from evaluate import macro_f_beta
from features import FEATURE_COLS
from score import build_features


def load_ground_truth(path: str) -> dict:
    truth = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1 = row[0]
            ids = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
            truth[s1] = ids
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../../dataset")
    ap.add_argument("--model-out", default="../model")
    ap.add_argument("--sample-frac", type=float, default=1.0, help="subsample S1 train rows for a fast dry run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(args.model_out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[train] loading sources...")
    s1 = load_source(os.path.join(args.data_dir, "train", "train_source1.tsv")).rename(
        columns={"entity_id": "source1_entity_id"}
    )
    s2 = load_source(os.path.join(args.data_dir, "train", "train_source2.tsv"))
    s3 = load_source(os.path.join(args.data_dir, "train", "train_source3.tsv"))
    truth = load_ground_truth(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))

    if args.sample_frac < 1.0:
        s1 = s1.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
        print(f"[train] sampled S1 down to {len(s1)} rows")

    print("[train] blocking...")
    pairs = generate_candidates(s1, s2, s3)
    print(f"[train] {len(pairs)} candidate pairs for {s1['source1_entity_id'].nunique()} S1 rows")

    # --- recall ceiling: what fraction of ground-truth matches survived blocking ---
    s1_id_set = set(s1["source1_entity_id"])
    sampled_truth = {k: v for k, v in truth.items() if k in s1_id_set}
    total_true = sum(len(v) for v in sampled_truth.values())
    cand_by_s1 = pairs.groupby("source1_entity_id")["cand_id"].apply(set)
    recovered = sum(
        len(v & cand_by_s1.get(k, set())) for k, v in sampled_truth.items()
    )
    recall_ceiling = recovered / total_true if total_true else 1.0
    n_zero_cand = len(s1_id_set) - cand_by_s1.index.nunique()
    cand_sizes = cand_by_s1.map(len)
    print(f"[train] blocking recall ceiling: {recovered}/{total_true} = {recall_ceiling:.4f}")
    print(f"[train] candidates/entity: mean={len(pairs)/len(s1_id_set):.1f} "
          f"median={cand_sizes.median():.0f} max={cand_sizes.max()}; "
          f"{n_zero_cand} entities got zero candidates")

    print("[train] attaching features...")
    s1_side, cand_side = build_side_lookups(s1, s2, s3)
    feats = build_features(pairs, s1_side, cand_side)
    pairs = pd.concat(
        [pairs[["source1_entity_id", "cand_id"]].reset_index(drop=True), feats], axis=1
    )

    pairs["label"] = [
        1 if cid in sampled_truth.get(s1id, ()) else 0
        for s1id, cid in zip(pairs["source1_entity_id"], pairs["cand_id"])
    ]

    # --- split by S1 id, not by row, to avoid leakage ---
    # Split over EVERY sampled S1 entity, not just those that got candidates: an
    # entity blocking found nothing for still gets scored by the metric (1.0 if it
    # truly has no matches, 0.0 otherwise). Drawing the split from `pairs` would
    # silently exclude them and bias both the threshold and the reported F0.5.
    unique_s1 = np.array(sorted(s1_id_set), dtype=object)
    rng.shuffle(unique_s1)
    n_val = max(1, int(len(unique_s1) * args.val_frac))
    val_ids = set(unique_s1[:n_val])
    is_val = pairs["source1_entity_id"].isin(val_ids)
    train_pairs, val_pairs = pairs.loc[~is_val], pairs.loc[is_val]
    print(f"[train] train pairs {len(train_pairs)} (pos {train_pairs.label.sum()}), "
          f"val pairs {len(val_pairs)} (pos {val_pairs.label.sum()})")

    dtrain = xgb.DMatrix(train_pairs[FEATURE_COLS], label=train_pairs["label"])
    dval = xgb.DMatrix(val_pairs[FEATURE_COLS], label=val_pairs["label"])
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": args.seed,
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20, verbose_eval=50,
    )

    val_pairs = val_pairs.copy()
    val_pairs["prob"] = booster.predict(dval)

    # required val S1 ids = ALL true S1 ids in val split (includes zero-candidate singletons)
    val_truth = {k: v for k, v in sampled_truth.items() if k in val_ids}

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        kept = val_pairs.loc[val_pairs["prob"] >= t]
        pred_map = kept.groupby("source1_entity_id")["cand_id"].apply(set).to_dict()
        score = macro_f_beta(pred_map, val_truth)
        if score > best_score:
            best_score, best_t = score, float(t)
    print(f"[train] best threshold={best_t:.2f} val macro F0.5={best_score:.4f}")

    booster.save_model(os.path.join(args.model_out, "model.json"))
    meta = {
        "feature_cols": FEATURE_COLS,
        "threshold": best_t,
        "val_macro_f0_5": best_score,
        "blocking_recall_ceiling": recall_ceiling,
        "n_train_pairs": int(len(train_pairs)),
        "n_val_pairs": int(len(val_pairs)),
    }
    with open(os.path.join(args.model_out, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[train] saved model + meta to", args.model_out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
