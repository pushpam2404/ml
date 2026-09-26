"""Run blocking + the trained classifier on the test set and write outputs.

Usage:
    python3 predict.py --data-dir ../../../dataset --model-dir ../model \
        --output-dir ../../../output
"""
import argparse
import json
import os
import sys

import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
from blocking import generate_candidates
from data_io import load_source, build_side_lookups
from score import predict_proba
from selection import select_by_threshold, select_expected_f
from write_outputs import write_id_list_tsv, pairs_to_id_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../../dataset")
    ap.add_argument("--model-dir", default="../model")
    ap.add_argument("--output-dir", default="../../../output")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.model_dir, "model_meta.json")) as f:
        meta = json.load(f)
    feature_cols, threshold = meta["feature_cols"], meta["threshold"]
    booster = xgb.Booster()
    booster.load_model(os.path.join(args.model_dir, "model.json"))

    print("[predict] loading test sources...")
    s1 = load_source(os.path.join(args.data_dir, "test", "test_source1.tsv")).rename(
        columns={"entity_id": "source1_entity_id"}
    )
    s2 = load_source(os.path.join(args.data_dir, "test", "test_source2.tsv"))
    s3 = load_source(os.path.join(args.data_dir, "test", "test_source3.tsv"))
    required_s1_ids = s1["source1_entity_id"].tolist()

    print("[predict] blocking...")
    pairs = generate_candidates(s1, s2, s3)
    n_candidates = len(pairs)
    print(f"[predict] {n_candidates} candidate pairs for {len(required_s1_ids)} S1 rows "
          f"({n_candidates / len(required_s1_ids):.1f} avg candidates/entity)")

    candidate_id_map = pairs_to_id_map(pairs)
    write_id_list_tsv(
        os.path.join(args.output_dir, "candidate_pairs.tsv"),
        required_s1_ids, candidate_id_map, "candidate_entity_ids",
    )

    print("[predict] scoring candidates...")
    s1_side, cand_side = build_side_lookups(s1, s2, s3)
    pairs = pairs.copy()
    pairs["prob"] = predict_proba(booster, pairs, s1_side, cand_side, feature_cols)

    # Apply the same rule training measured as best, via the same code path.
    rule = meta.get("selection_rule", "threshold")
    if rule == "expected_f":
        kept = select_expected_f(pairs)
    else:
        kept = select_by_threshold(pairs, threshold)
    print(f"[predict] selection rule: {rule} -> {len(kept)} matches kept "
          f"of {len(pairs)} candidates")
    match_id_map = pairs_to_id_map(kept)
    write_id_list_tsv(
        os.path.join(args.output_dir, "matching_results.tsv"),
        required_s1_ids, match_id_map, "matched_entity_ids",
    )
    n_matched = sum(1 for v in match_id_map.values() if v)
    print(f"[predict] wrote outputs to {args.output_dir}; "
          f"{n_matched}/{len(required_s1_ids)} S1 entities got >=1 match "
          f"(threshold={threshold})")


if __name__ == "__main__":
    main()
