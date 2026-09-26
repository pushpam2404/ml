"""Turning per-pair match probabilities into a predicted match set per entity.

The competition metric is macro-averaged F_0.5: computed per Source-1 entity, then
averaged. A single global probability threshold ignores that structure -- it judges an
entity with four plausible matches by the same cutoff as one with a single confident
match. Both rules below are implemented here and used by training and inference alike,
so the rule that is tuned is the rule that runs.
"""
import numpy as np
import pandas as pd

BETA2 = 0.25  # beta^2 for F_0.5


def _f_beta(precision, recall):
    """Vectorized F_0.5; 0 where precision and recall are both 0."""
    denom = BETA2 * precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (1 + BETA2) * precision * recall / denom
    return np.where(denom > 0, out, 0.0)


def select_by_threshold(pairs: pd.DataFrame, threshold: float,
                        prob_col: str = "prob") -> pd.DataFrame:
    """Keep every pair scoring at or above a single global threshold."""
    return pairs.loc[pairs[prob_col].to_numpy() >= threshold]


PROB_FLOOR = 0.02  # candidates below this are dropped before the per-entity maths:
                    # they can never be in the selected prefix, and contribute
                    # negligibly to the expected-match-count and empty-probability
                    # sums, but at test scale they are ~90% of the rows.


def select_expected_f(pairs: pd.DataFrame, prob_col: str = "prob",
                      entity_col: str = "source1_entity_id",
                      prob_floor: float = PROB_FLOOR) -> pd.DataFrame:
    """Per entity, keep the prefix of highest-scoring candidates that maximises
    *expected* F_0.5 under the model's own probabilities.

    For an entity whose candidates have probabilities p1 >= p2 >= ... , taking the top
    n gives an expected true-positive count of sum(p_i, i<=n), so
        precision_hat = sum(p_i, i<=n) / n
        recall_hat    = sum(p_i, i<=n) / sum(all p_i)
    and we pick the n maximising F_0.5 of those. Predicting nothing is also a
    candidate decision, worth the probability that the entity genuinely has no match
    (prod(1 - p_i)) since an empty prediction scores a full 1.0 when correct -- which
    is how this rule earns credit on singletons instead of guessing on them.

    Fully vectorized: one sort plus group-wise cumulative sums, no python loop per
    entity (there are ~1.7M of them at test time).
    """
    if pairs.empty:
        return pairs

    pairs = pairs.loc[pairs[prob_col].to_numpy() >= prob_floor]
    if pairs.empty:
        return pairs

    df = pairs.sort_values([entity_col, prob_col], ascending=[True, False],
                           kind="stable").reset_index(drop=True)
    p = np.clip(df[prob_col].to_numpy(dtype="float64"), 1e-9, 1 - 1e-9)
    grp = df.groupby(entity_col, sort=False)

    n = grp.cumcount().to_numpy() + 1                      # 1..k within entity
    cum = np.asarray(grp[prob_col].cumsum(), dtype="float64")
    total = np.asarray(grp[prob_col].transform("sum"), dtype="float64")

    f_at_n = _f_beta(cum / n, np.divide(cum, total, out=np.zeros_like(cum),
                                        where=total > 0))

    # best prefix length per entity
    df["_f"] = f_at_n
    df["_n"] = n
    best = df.loc[df.groupby(entity_col, sort=False)["_f"].idxmax(), [entity_col, "_f", "_n"]]

    # value of predicting nothing at all: P(entity truly has no match)
    log1m = pd.Series(np.log1p(-p), index=df.index)
    p_empty = np.exp(log1m.groupby(df[entity_col], sort=False).sum())
    best = best.merge(p_empty.rename("_p_empty"), left_on=entity_col, right_index=True,
                      how="left")

    keep_n = np.where(best["_p_empty"].to_numpy() > best["_f"].to_numpy(),
                      0, best["_n"].to_numpy())
    cutoff = pd.Series(keep_n, index=best[entity_col].to_numpy())

    limit = df[entity_col].map(cutoff).to_numpy()
    return df.loc[n <= limit].drop(columns=["_f", "_n"])


def demo():
    # one entity with two confident matches and one junk candidate; the junk one
    # should be dropped while both confident ones are kept
    pairs = pd.DataFrame({
        "source1_entity_id": ["S1-1"] * 3 + ["S1-2"] * 2,
        "cand_id": ["S2-1", "S3-1", "S2-9", "S2-5", "S3-5"],
        "prob": [0.97, 0.93, 0.02, 0.03, 0.01],
    })
    got = select_expected_f(pairs)
    kept1 = set(got.loc[got.source1_entity_id == "S1-1", "cand_id"])
    assert kept1 == {"S2-1", "S3-1"}, kept1
    # S1-2 has only weak candidates -> predicting empty is worth more than guessing
    kept2 = set(got.loc[got.source1_entity_id == "S1-2", "cand_id"])
    assert kept2 == set(), kept2

    # a lone strong candidate must still be kept
    single = pd.DataFrame({
        "source1_entity_id": ["S1-3"],
        "cand_id": ["S2-7"],
        "prob": [0.99],
    })
    assert set(select_expected_f(single)["cand_id"]) == {"S2-7"}

    # threshold rule sanity
    assert set(select_by_threshold(pairs, 0.5)["cand_id"]) == {"S2-1", "S3-1"}
    # matches the worked example in the problem statement: P=2/3, R=1 -> 0.714
    assert abs(_f_beta(np.array([2 / 3]), np.array([1.0]))[0] - 0.7143) < 1e-3
    print("selection.py demo OK")


if __name__ == "__main__":
    demo()
