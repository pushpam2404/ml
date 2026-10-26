# Business entity resolution (ML Challenge 2026)

![tests](https://github.com/pushpam2404/ml/actions/workflows/ci.yml/badge.svg)

My entry for the ML Challenge 2026 problem "Business Entity Resolution". Given business records from three sources with noisy names and addresses, find which Source 2 and Source 3 records describe the same business as each Source 1 record. Source 1 is the deduplicated reference. A Source 1 entity can match zero, one or many records in the other two sources, and there is no shared identifier.

The metric is macro-averaged F0.5 per Source 1 entity, so a false merge costs about twice as much as a missed match, and correctly returning an empty list for a singleton scores 1.0. The full problem statement, as given by the organisers, is in [student_resource/README.md](student_resource/README.md). The test set has a country (France) that does not appear in training, so nothing in the code keys on a fixed list of countries.

## Approach

Comparing all pairs is around 10^13 comparisons at the challenge's scale (about 2.2M / 5.0M / 5.3M training records in sources 1 / 2 / 3), so there are two stages.

1. **Blocking.** Records are only compared inside the same country. This is lossless on the training data: none of the 7.6M ground-truth pairs crosses a country. Candidates come from four inverted-index passes over the Source 2 and Source 3 records, each ranked by summed IDF of the shared keys and cut to a top-K per Source 1 entity:
   - name tokens, address tokens and Metaphone codes of name tokens (top 40)
   - address tokens only (top 20)
   - PIN code plus name tokens (top 20)
   - character 4-grams of the name, for typos and glued names like `wilfordhancock.com` (top 15)

   Passes are unioned. Chunks of Source 1 are processed in parallel where `fork` is safe (not on macOS).
2. **Matching.** 21 string-similarity features per candidate pair (token Jaccard, character trigram Jaccard, rapidfuzz ratios, Jaro-Winkler, Levenshtein, PIN and house-number agreement, phonetic match and so on; see `src/features.py`). An XGBoost classifier scores each pair (depth 7, eta 0.05, up to 600 rounds, early stopping on AUCPR).
3. **Selection.** Two ways of turning probabilities into a match list are implemented in `src/selection.py`: one global probability threshold, and a per-entity rule that keeps the prefix of candidates maximising expected F0.5 (and can choose to return nothing). Training measures both on a held-out split by Source 1 id and keeps the better one.

## Results

What I can show from the files in the repo is the final model's own record, `student_resource/code/business_entity_resolution/model/model_meta.json`, produced by a training run on a 10% sample of Source 1 (220,682 entities) blocked against the full Source 2 and 3:

| | |
|---|---|
| Blocking recall ceiling | 0.8470 |
| Candidate pairs | 27.7M train, 3.1M validation |
| Validation macro F0.5 | 0.8512 (global threshold 0.65; the per-entity rule scored 0.8510) |

The earlier models, for scale. All numbers are validation macro F0.5 from each run's `model_meta.json`; the runs used different Source 1 sample sizes, so the pair counts differ.

| Run | Features | Blocking recall | Validation F0.5 |
|---|---|---|---|
| first model | 10 | 0.6847 | 0.7028 (threshold), 0.7082 (per-entity rule) |
| tuning run, half the pairs of the first | 10 | 0.6847 | 0.7055 |
| second model, sample | 21 | 0.8380 | 0.8406 |
| third model, sample | 21 | 0.8199 | 0.8317 |
| **final** | 21 | 0.8470 | **0.8512** |

The first model scored 0.711 on the public leaderboard when I uploaded it. I did not record the leaderboard score of the final model, so there is no leaderboard number for it here. The challenge data is not in this repository, so I could not reproduce any of these numbers when cleaning the repo up. What I did run is described under Tests.

The methodology write-up I submitted is [student_resource/Documentation_template.md](student_resource/Documentation_template.md). Its sections 3 to 5 describe the first model; the note at the top says which parts are out of date.

## Layout

```
student_resource/
  README.md                          problem statement (from the organisers)
  Documentation_template.md          methodology write-up
  utils/validate_submission.py       organisers' format checker (stdlib only)
  package_submission.sh              builds the submission zip
  code/business_entity_resolution/
    requirements.txt                 pinned dependencies
    model/                           final XGBoost model + metadata (7 MB)
    src/                             pipeline: normalize, blocking, features, score,
                                     selection, train, predict, evaluate, tune_selection
    tests/                           unit tests and a synthetic end-to-end run
```

## Setup

Python 3.12 with the pinned requirements:

```bash
cd student_resource/code/business_entity_resolution
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The data is not included. The files come from the challenge portal and I have no public URL to script a download from. Put them in `student_resource/dataset/` with this layout (all tab-separated):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## Usage

Run from `student_resource/code/business_entity_resolution/src/`.

```bash
# train on the training set, writes ../model/model.json and model_meta.json
../.venv/bin/python train.py --data-dir ../../../dataset --model-out ../model
# train.py --sample-frac 0.01 does a quick dry run on 1% of Source 1

# predict with the committed (or freshly trained) model
../.venv/bin/python predict.py --data-dir ../../../dataset --model-dir ../model --output-dir ../../../output

# or both steps
../.venv/bin/python run_pipeline.py --data-dir ../../../dataset --model-out ../model --output-dir ../../../output

# check the files before uploading
cd ../../.. && python3 utils/validate_submission.py \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

`predict.py` needs only the test files and a model directory. At full scale it generated about 236M candidate pairs for 1.73M test entities, so expect hours and tens of GB of RAM. Training re-creates `model/val_predictions.parquet`, which `tune_selection.py` uses to compare selection rules in seconds. That file is not committed.

## Tests

```bash
cd student_resource/code/business_entity_resolution
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
```

21 tests, about 6 seconds. They cover the metric against the worked example in the problem statement, normalisation, both selection rules, each module's own self-check, and a full train, predict and validate run on a small made-up dataset (`tests/synthetic.py`). That run includes a country absent from training and has to pass the organisers' validator. It proves the pipeline runs and writes a valid submission. It says nothing about accuracy on the real data, because the made-up data is far easier. I also checked that the committed `model/` loads and scores with the pinned xgboost.

## Limitations

- The `&` to `and` entry in the name synonym table never fires, because punctuation is stripped before synonyms are applied. "A & B" and "A and B" therefore normalise differently. Fixing it changes the features, so the committed model should be retrained afterwards.
- `MIN_PAIR_SCORE` in `blocking.py` is an absolute cut-off on a score that grows with corpus size, so it was calibrated for the full challenge data and rejects almost everything on a tiny corpus. The self-check switches it off for that reason.
- `jellyfish==1.0.0` had no installable wheel for my Python 3.12, so `requirements.txt` pins 1.0.3. Metaphone feeds one feature and one blocking pass. I could not confirm that 1.0.3 produces the same codes as the version the model was trained with.
- Blocking recall is the ceiling: at 0.847, about 15% of true matches never reach the classifier.
- The test-set run produced 236M candidate pairs, roughly 136 per entity. The organisers analyse the candidate file (recall ceiling, reduction ratio), so a smaller set is better and this one is large.

## License

MIT, see [LICENSE](LICENSE). The problem statement and `utils/validate_submission.py` are the organisers' files.
