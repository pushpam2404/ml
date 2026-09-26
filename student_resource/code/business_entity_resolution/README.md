# Business Entity Resolution — Pipeline

Blocking (inverted-token-index, country-scoped) + XGBoost tabular classifier
over hand-engineered name/address similarity features. See
`../../Documentation_template.md` (repo root of the zip) for methodology,
metrics, and error analysis.

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Reproduce end-to-end

Run from `code/business_entity_resolution/src/`, assuming the challenge's
`dataset/` folder sits at the repo root next to `output/` (adjust `--data-dir`
/ `--output-dir` if your layout differs):

```bash
cd src
../.venv/bin/python train.py --data-dir ../../../dataset --model-out ../model
../.venv/bin/python predict.py --data-dir ../../../dataset --model-dir ../model \
    --output-dir ../../../output
```

Or both steps at once:

```bash
../.venv/bin/python run_pipeline.py --data-dir ../../../dataset --model-out ../model
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Validate before submitting:

```bash
cd ../../..   # student_resource/
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`train.py --sample-frac 0.01` runs a fast dry run on a 1% S1 subsample for
development iteration.

## Source layout

- `src/normalize.py` — text normalization: ASCII-fold (unidecode) + lowercase,
  legal-suffix/address-abbreviation synonym folding, PIN / house-number
  extraction, tokenization.
- `src/blocking.py` — country-scoped inverted-token-index candidate
  generation with common-token pruning and top-K capping per S1 entity. This
  is the exact code path that produces `candidate_pairs.tsv`.
- `src/features.py` — name/address similarity features (Jaccard, character
  trigram Jaccard, rapidfuzz ratios, PIN/house-number exact match).
- `src/data_io.py` — TSV loading + assembling feature-ready pair frames.
- `src/evaluate.py` — macro-averaged F_0.5, identical to the leaderboard's
  scoring formula, used to pick the classification threshold.
- `src/train.py` — full training pipeline: block, label from ground truth,
  split by S1 id, train XGBoost, tune the threshold, save `model.json` +
  `model_meta.json`.
- `src/predict.py` — block the test set, score with the trained model, write
  both output TSVs.
- `src/run_pipeline.py` — chains train.py then predict.py.

Every module also has a `python3 <module>.py` self-check (`demo()` +
`assert`s) for a quick correctness smoke test in isolation.
