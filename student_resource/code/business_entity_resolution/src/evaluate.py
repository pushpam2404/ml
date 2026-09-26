"""Macro-averaged F_0.5 scoring, matching the leaderboard's metric exactly."""


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(predicted: dict, truth: dict, beta: float = 0.5) -> float:
    """predicted/truth: {source1_entity_id: set(matched_ids)}. Every truth key scored."""
    scores = []
    for s1, true_ids in truth.items():
        pred_ids = predicted.get(s1, set())
        if not true_ids and not pred_ids:
            scores.append(1.0)
            continue
        if not pred_ids:
            scores.append(0.0)
            continue
        tp = len(true_ids & pred_ids)
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids) if true_ids else 0.0
        scores.append(f_beta(precision, recall, beta))
    return sum(scores) / len(scores) if scores else 0.0


def demo():
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    pred = {"S1-1": {"S2-1", "S2-2", "S3-1"}, "S1-2": set()}
    score = macro_f_beta(pred, truth)
    # S1-1: precision=2/3, recall=1.0 -> 0.714..., S1-2: 1.0 -> avg ~0.857
    assert abs(score - 0.857) < 0.01, score
    print("evaluate.py demo OK, score=", score)


if __name__ == "__main__":
    demo()
