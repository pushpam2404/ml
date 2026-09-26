"""Compare selection rules offline against cached validation scores.

train.py writes model/val_predictions.parquet; this scores candidate decision rules
against it in seconds, so the rule can be explored without repeating training.

    python3 tune_selection.py --model-dir ../model --data-dir ../../../dataset

Reports macro F_0.5 exactly as the leaderboard computes it. Note the ground truth is
needed to recover entities that got NO candidates: they are absent from the
predictions file but are still scored (1.0 if genuinely singletons, else 0.0), and
leaving them out would flatter every rule equally but misleadingly.
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import macro_f_beta
from selection import select_by_threshold, select_expected_f


def load_truth(path):
    truth = {}
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            truth[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1].strip() else set()
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="../model")
    ap.add_argument("--data-dir", default="../../../dataset")
    args = ap.parse_args()

    val = pd.read_parquet(os.path.join(args.model_dir, "val_predictions.parquet"))
    truth_all = load_truth(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))

    # The validation split covers every S1 entity whose pairs are here, PLUS entities
    # that got no candidates at all. We can only recover the former from this file, so
    # report on exactly those entities and say so.
    val_ids = set(val["source1_entity_id"].unique())
    val_truth = {k: truth_all.get(k, set()) for k in val_ids}
    print(f"{len(val)} validation pairs over {len(val_ids)} entities "
          f"({sum(1 for v in val_truth.values() if not v)} true singletons)", flush=True)

    def score_of(kept):
        return macro_f_beta(
            kept.groupby("source1_entity_id")["cand_id"].apply(set).to_dict(), val_truth
        )

    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        rows.append((f"threshold {t:.2f}", score_of(select_by_threshold(val, float(t)))))
    best_thresh = max(rows, key=lambda r: r[1])

    rows.append(("per-entity expected-F", score_of(select_expected_f(val))))

    # does sharpening/softening the probabilities help the expected-F maths?
    for alpha in (0.5, 1.5, 2.0):
        adj = val.copy()
        adj["prob"] = adj["prob"].to_numpy() ** alpha
        rows.append((f"expected-F (p^{alpha})", score_of(select_expected_f(adj))))

    # hybrid: expected-F, but never return empty when a decent candidate exists
    for floor in (0.35, 0.5):
        kept = select_expected_f(val)
        covered = set(kept["source1_entity_id"].unique())
        rescue = val.loc[
            (~val["source1_entity_id"].isin(covered)) & (val["prob"] >= floor)
        ]
        rescue = rescue.sort_values("prob", ascending=False).groupby(
            "source1_entity_id", sort=False).head(1)
        rows.append((f"expected-F + top1 rescue>={floor}",
                     score_of(pd.concat([kept, rescue], ignore_index=True))))

    print()
    print(f"{'rule':38s} {'val macro F0.5':>14s}")
    for name, s in rows:
        mark = "  <-- best threshold" if (name, s) == best_thresh else ""
        print(f"{name:38s} {s:14.4f}{mark}")
    print()
    best = max(rows, key=lambda r: r[1])
    print(f"BEST: {best[0]} at {best[1]:.4f}")


if __name__ == "__main__":
    main()
