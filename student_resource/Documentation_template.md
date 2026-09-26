# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

> **STATUS: DRAFT — metric fields marked `TBD` are not yet filled.** Numbers will be
> taken from the pipeline's own output (`train.py` prints the blocking recall
> ceiling, candidate-set statistics, the tuned threshold and the validation macro
> F_0.5, and writes them to `code/business_entity_resolution/model/model_meta.json`).
> No number in this document is to be written by hand.

---

## 1. Executive Summary

Country-partitioned inverted-index blocking with IDF-weighted candidate ranking,
followed by an XGBoost classifier over string-similarity features and a single
probability threshold tuned directly against the competition's macro F_0.5 metric.
The two things that mattered most were not model choice: they were (a) discovering
that an aggressive blocking token-frequency cutoff was silently destroying most of
the achievable recall, and (b) ranking candidates by summed IDF rather than by raw
count of shared tokens, which raised recall substantially at *zero* increase in
candidate-set size.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from EDA on the provided training data:

- **Country never crosses in the ground truth.** All 7,638,365 training match pairs
  were checked: 0 crossed a country boundary. Country is therefore a free, exactly
  lossless blocking partition, and it is applied as an open set of string labels
  (never hard-coded to `{US, India}`), so the test set's unseen `France` flows
  through unchanged.
- **Scale.** 2.21M Source-1 / 5.03M Source-2 / 5.29M Source-3 training records;
  1.73M / 4.89M / 5.08M at test. An all-pairs comparison is ~10^13 pairs, so
  blocking is not optional.
- **Match multiplicity.** 123,247 of 2.21M training Source-1 entities (5.6%) are
  singletons with zero matches; the rest mostly have 2-8 matches. Since a correctly
  predicted empty list scores a full 1.0 and any false merge on a singleton scores
  0.0, singletons are a meaningful share of the metric.
- **Missing data.** `business_name` is never empty; ~3% of Source-2/3 records have
  an empty `business_address`, so the pipeline must not assume address is present.
- **Observed noise patterns.** Legal-suffix variants (Pvt/Private, Ltd/Limited);
  word-order transposition (`orion organic ltd` vs `ltd orion organic`);
  character-level typos (`organic` vs `ornanac`); domain-style trade names
  (`wilfordhancock.com` for "Wilford Hancock"); cross-script transliteration for
  India records, where the name appears in Devanagari/Bengali/Kannada while the
  address usually stays in Latin script; abbreviated street and state names; and
  off-by-one house numbers.
- **Degenerate high-frequency tokens.** Transliteration artefacts of legal suffixes
  (`limittedd` appearing 308k times, `praaivett` 176k) dominate token frequency
  while carrying no identifying signal, and must be separated from genuinely common
  but still-useful words (`pizza` ~5.6k, `global` ~22k).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two stage)
**Core Innovation:** IDF-weighted inverted-index ranking, which improves the recall
ceiling *without* enlarging the candidate set — the axis the challenge scores
separately. Combined with a calibrated (measured, not guessed) token-frequency
cutoff.

---

## 3. Candidate Generation (Blocking)

- **Partition:** exact `country` match, justified empirically (see 2.1) as lossless.
- **Blocking keys**, all within a country partition:
  - significant `business_name` tokens (normalized, legal-suffix stopwords removed),
  - `business_address` tokens (street/city/state words, abbreviations canonicalized),
  - the postal/PIN code extracted from the raw address.
- **Ranking:** candidates are scored by the **sum of IDF weights of shared keys**,
  and the top-K per Source-1 entity per source is kept. Each field is weighted with
  its own IDF, so a token that is common in addresses but rare in names is weighted
  correctly in each context.
- **Frequency pruning:** a (country, token) key indexing more than `MAX_TOKEN_DF`
  candidates is dropped. This is a *compute* guard, not a quality mechanism —
  verified by measurement: raising the cutoff from 3,000 to 10,000 produced an
  identical recall, because IDF weighting already discounts common tokens.
- **Fallback:** entities with no candidates from the main pass get a relaxed
  address-only pass. Measured: this affects ~1.6% of entities, and widening it
  changed recall by 0.0000 — the binding constraint was ranking, not coverage.
- **Determinism:** scores tie frequently at the top-K boundary, so selection breaks
  ties on `cand_id`. Without a total order, results depended on row processing order
  and varied with batch size; the run is now reproducible.

**How true matches were kept:** by measuring the recall ceiling directly on held-out
training data at every change, rather than assuming. The decisive measurements:

| Configuration | Blocking recall | Candidates / entity |
| --- | --- | --- |
| Raw-count ranking, `MAX_TOKEN_DF`=200 | 0.3876 | 57 |
| Raw-count ranking, `MAX_TOKEN_DF`=3000 | 0.6164 | 60 |
| **IDF ranking, `MAX_TOKEN_DF`=3000 (chosen)** | **0.6792** | **60** |
| IDF ranking, `MAX_TOKEN_DF`=10000 | 0.6792 | 60 |
| IDF ranking, TOP_K=60 | 0.7267 | 120 |

The first row was the original configuration: a cutoff of 200 pruned *every* token
of a common-word business name (e.g. "Straight Edge Pizza"), leaving such entities
with no usable candidates at all. Raising the cutoff and switching to IDF ranking
nearly doubled recall while holding the candidate set at 60/entity. TOP_K=60 buys
further recall but doubles the candidate set, which the challenge penalises.

- **Candidate pairs generated (test):** TBD
- **Candidates per Source-1 entity (test):** TBD
- **Blocking recall ceiling (held-out train):** TBD (full-scale run)

`candidate_pairs.tsv` is written from the exact dataframe that is then fed to the
model — one code path, so the audited candidate set cannot diverge from what
inference actually scored.

---

## 4. Matching Model

**Features used** (computed per candidate pair):

- *Name:* token-set Jaccard, character-trigram Jaccard, `token_sort_ratio`
  (transposition-tolerant), `partial_ratio` (substring/DBA-tolerant), absolute
  length difference.
- *Address:* token-set Jaccard, full-string similarity ratio, exact PIN-code match,
  exact house-number match.
- *Structural:* whether either side's address was empty.
- Deliberately **no** `country_match` feature: blocking guarantees it is always 1, so
  it would be a constant.

**Model type:** XGBoost (`binary:logistic`, depth 5, eta 0.1, up to 300 rounds with
early stopping on validation AUCPR). Apache-2.0 licensed and far below the 8B
parameter limit. Positives are ground-truth matches surviving blocking; negatives
are the remaining same-entity candidates, i.e. naturally hard negatives, so no
random negative sampling is needed.

**Threshold selection:** a single global probability threshold, grid-searched to
maximise **macro F_0.5 computed exactly as the leaderboard defines it** (per entity,
then averaged, singletons included). The split is by Source-1 entity id — never by
row — so no entity's pairs appear on both sides. Critically, the validation
population includes entities for which blocking found *nothing*: they are still
scored by the metric (1.0 if truly singletons, 0.0 otherwise), and excluding them
would bias both the threshold and the reported score.

- **Tuned threshold:** TBD
- **Validation macro F_0.5:** TBD

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, held-out validation):** TBD
- **Blocking recall ceiling:** TBD — this is the dominant constraint on the score.
  With perfect precision, a recall of *r* caps per-entity F_0.5 at
  `1.25r / (0.25 + r)`, so lifting blocking recall is worth more than any further
  classifier tuning.
- **Common false negatives (missed matches):** dominated by pairs lost at blocking
  rather than rejected by the classifier. Categories identified: cross-script
  transliteration where the phonetic Latin form shares no whole token with the
  English name (`vijan`/`vision`), domain-style names that tokenize as one blob
  (`wilfordhancock.com`), and character-level typos.
- **Common false positives (wrong merges):** different businesses sharing an address
  (same building/plaza) with generic names, where address evidence is strong and
  name evidence is weak.

### Known limitations / next steps

1. **Character n-gram blocking keys**, to catch typos and domain-style names that
   whole-word matching cannot see.
2. **Cosine-normalised scoring** instead of a raw IDF sum. A raw sum rewards
   candidates with many matching tokens, which may bias toward records with long
   addresses (common for India); normalising by the candidate's IDF norm is the
   standard correction.
3. Trained on a 10% sample of Source-1 entities — ample for a ~10-feature model, but
   the full set is available if the model proves data-limited.

---

## 6. Conclusion

The score in this task is gated by candidate generation, not by classifier
sophistication: every large gain came from measuring *why* true matches were being
lost and fixing the blocking stage, while the deliberately simple tabular model was
never the bottleneck. The most valuable engineering habit was refusing to tune
parameters by intuition — the token-frequency cutoff that looked reasonable at 200
was destroying 44% of the achievable recall, and only direct measurement revealed it.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`, with all source under `src/` and exact
reproduction commands in its `README.md`:

- `normalize.py` — ASCII folding (transliteration), legal-suffix and address
  abbreviation canonicalization, PIN/house-number extraction, tokenizers.
- `blocking.py` — country-partitioned IDF-weighted inverted-index candidate
  generation. Produces `candidate_pairs.tsv`.
- `features.py` — pair similarity features.
- `score.py` — chunked feature building and scoring, shared by training and
  inference so the two cannot drift apart.
- `data_io.py`, `write_outputs.py` — TSV I/O and submission formatting.
- `evaluate.py` — macro F_0.5, identical to the leaderboard formula.
- `train.py` / `predict.py` / `run_pipeline.py` — entry points.

Every module has a runnable `python3 <module>.py` self-check.

**Engineering notes that materially affected results:**

- pandas 3.0 defaults string columns to an Arrow-backed dtype, which makes
  element-wise Python string operations pay a per-element scalar conversion. On
  10M-row sources this made the pipeline ~30x slower until the legacy
  numpy-object dtype was restored; diagnosed by stack profiling, not guesswork.
- Both the blocking join and feature hydration materialise intermediates that scale
  with input size (billions of rows / far beyond memory at full scale). Both now
  stream in bounded chunks, verified to produce results identical to a single-shot
  run.
- Prepped source frames were trimmed from 3.9GB to 1.9GB each by dropping raw text
  after normalization and deriving token lists on demand instead of storing them as
  per-row Python objects.

### B. Additional Results

TBD — blocking recall ceiling, candidate-set size distribution, and validation
macro F_0.5 from the full-scale run.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
