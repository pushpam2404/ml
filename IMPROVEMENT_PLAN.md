# Score Improvement Plan: 0.711 → 0.95+

> **Current leaderboard score: 0.711** | Top teams: 0.990 | Rank: ~3000
> **Metric:** Macro-averaged F₀.₅ (precision-heavy — false merges cost more than missed matches)
> **Time remaining:** Check the leaderboard timer — 72-hour hackathon

---

## Gap Analysis — Why We Score 0.711 and They Score 0.990

| Component | Our Current State | What Top Teams Likely Do | Impact |
|---|---|---|---|
| **Blocking recall** | **0.6847** — 31.5% of true matches never enter the candidate set | **>0.97** — near-perfect recall via multi-strategy blocking | 🔴 CRITICAL |
| **Candidate set size** | 60/entity (TOP_K=30 per source) | 200-500/entity — they accept bigger sets and let the model filter | 🔴 HIGH |
| **Blocking keys** | Whole-word tokens only | Character n-grams + phonetic keys + multiple passes | 🔴 HIGH |
| **French legal suffixes** | Not handled — SARL/SAS/EURL/SASU not in synonym map | Folded like Pvt→pvt, Ltd→ltd | 🟡 MEDIUM |
| **Features** | 10 features (Jaccard, rapidfuzz ratios, PIN/house match) | 20-30+ features (edit distance, Jaro-Winkler, TF-IDF cosine, phonetic) | 🟡 MEDIUM |
| **Model** | XGBoost depth=5, 300 rounds | Deeper or ensemble, or same but on better features | 🟢 LOW |
| **Selection rule** | Per-entity expected-F (already good) | Similar — good selection helps ~0.005 | 🟢 LOW |

**The single biggest problem is blocking recall at 0.6847.** No downstream model can recover matches that were never generated as candidates. The theoretical max F₀.₅ at recall 0.6847 with perfect precision is ~0.92. The top teams at 0.99 must have blocking recall >0.97.

---

## Execution Plan — Ordered by Impact

### Priority Legend
- 🔴 **P0** = Do this FIRST, biggest score gain
- 🟡 **P1** = Do after P0, meaningful gain
- 🟢 **P2** = Nice to have, small gain

---

## P0-A: Increase TOP_K from 30 to 100 per source (5 min code change)

**Why:** TOP_K=30 per source (60 total) is extremely aggressive. The blocking recall measurement at TOP_K=60 showed recall jump from 0.685 → 0.727. At TOP_K=100, expect ~0.75-0.80+. The candidate set grows from ~60 to ~200 per entity which is still manageable.

**File:** `code/business_entity_resolution/src/blocking.py`

**Change line 19:**
```python
# BEFORE:
TOP_K = 30

# AFTER:
TOP_K = 100
```

**That's it.** One line. This alone could gain +0.04 to +0.08 on the leaderboard.

---

## P0-B: Lower MIN_PAIR_SCORE from 10.0 to 3.0 (5 min code change)

**Why:** MIN_PAIR_SCORE=10.0 pre-prunes candidate pairs before top-K selection. This was "calibrated" on the old corpus but is too aggressive — it can discard pairs that share a single moderately-rare token, which are exactly the hard cases we miss. Lowering to 3.0 keeps more marginal candidates in play for the model to judge.

**File:** `code/business_entity_resolution/src/blocking.py`

**Change line 22:**
```python
# BEFORE:
MIN_PAIR_SCORE = 10.0

# AFTER:
MIN_PAIR_SCORE = 3.0
```

---

## P0-C: Add Character N-Gram Blocking Keys (30-60 min code change)

**Why:** Whole-word token blocking completely misses:
- Typos: `organic` vs `ornanac` — zero shared whole tokens
- Domain-style names: `wilfordhancock.com` — tokenizes as one blob, shares nothing with "Wilford Hancock"
- Cross-script transliteration: `vijan` vs `vision` — phonetic Latin form shares no whole token
- Minor spelling variants: `theatre` vs `theater`

Character trigrams (`org`, `rga`, `gan`, `ani`, `nic`) overlap even with typos.

**File:** `code/business_entity_resolution/src/blocking.py`

**What to add:** A new function that generates character n-gram keys and a new index that gets concatenated with the existing token index.

Add this function after the existing `_token_index` function (around line 50):

```python
def _ngram_index(df: pd.DataFrame, id_col: str, text_col: str, n: int = 4) -> pd.DataFrame:
    """Generate character n-gram blocking keys from a text column.
    
    Character n-grams catch typos and partial matches that whole-word tokens miss entirely.
    Uses n=4 (4-grams) by default: short enough to match across typos,
    long enough to avoid ultra-common fragments.
    """
    def ngrams(text: str):
        # Remove spaces, generate unique n-grams
        s = text.replace(" ", "")
        if len(s) < n:
            return [s] if s else []
        return list(dict.fromkeys(s[i:i+n] for i in range(len(s) - n + 1)))
    
    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        id_col: df[id_col].to_numpy(),
        "token": df[text_col].map(ngrams).to_numpy(),
    }).explode("token")
    return long.dropna(subset=["token"])
```

**Then modify `block_one_source`** (around line 161) to include the n-gram index.

Find the lines where `full_index` is built (around line 177):
```python
    # BEFORE:
    full_index = pd.concat([name_idx, addr_idx, pin_idx], ignore_index=True)

    # AFTER — add character n-gram index for names:
    ngram_idx = _weighted_index(
        _ngram_index(coded, "cand_code", "name_norm", n=4),
        n_by_country
    )
    full_index = pd.concat([name_idx, addr_idx, pin_idx, ngram_idx], ignore_index=True)
    del ngram_idx
```

**And modify `_s1_long`** (around line 112) to also generate n-grams for S1:

```python
def _s1_long(s1: pd.DataFrame, fields=("name", "addr", "pin")) -> pd.DataFrame:
    """Build the long (country, token, source1_entity_id) frame for the S1 side."""
    parts = []
    if "name" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "name_norm", name_tokens))
        # Also add character n-gram keys for names
        parts.append(_ngram_index(s1, "source1_entity_id", "name_norm", n=4))
    if "addr" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "addr_norm", addr_tokens))
    if "pin" in fields:
        parts.append(
            s1.loc[s1["pin"] != "", ["country", "pin", "source1_entity_id"]].rename(
                columns={"pin": "token"}
            )
        )
    return pd.concat(parts, ignore_index=True)
```

> ⚠️ **Memory note:** Character n-grams generate more index entries than whole-word tokens. If memory is tight, use n=5 instead of n=4 (fewer n-grams per string, still catches most typos). Also increase `MAX_TOKEN_DF` from 3000 to 5000 to avoid pruning common n-grams that are still useful.

---

## P0-D: Add French Legal Suffix Normalization (10 min code change)

**Why:** France is **15% of the test set** (259,452 S1 entities) with zero training data. French business names use suffixes like SARL, SAS, EURL, SASU, SCI, S.A.S., S.A.R.L. that are the French equivalent of Pvt/Ltd. These are currently NOT normalized, so they pollute token matching and waste blocking slots. The current synonym map only handles English suffixes.

**File:** `code/business_entity_resolution/src/normalize.py`

**Expand the `_SYNONYMS` dictionary** (line 17):
```python
# BEFORE:
_SYNONYMS = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "llp": "llp", "llc": "llc", "company": "co",
    "and": "and", "&": "and",
}

# AFTER — add French legal suffixes:
_SYNONYMS = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "llp": "llp", "llc": "llc", "company": "co",
    "and": "and", "&": "and",
    # French legal suffixes (15% of test set is France)
    "sarl": "sarl", "sas": "sas", "eurl": "eurl", "sasu": "sasu",
    "sci": "sci", "snc": "snc", "sa": "sa",
    "societe": "soc", "société": "soc", "cie": "co",
    "groupe": "grp", "holding": "hldg",
    "fils": "fils", "freres": "freres", "frères": "freres",
    "associes": "assoc", "associés": "assoc",
}
```

**Also add French address synonyms** to `_ADDR_SYNONYMS` (line 24):
```python
# AFTER — add French address terms:
_ADDR_SYNONYMS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "apartment": "apt", "unit": "unit", "suite": "ste",
    # French address terms
    "rue": "rue", "allée": "allee", "allee": "allee",
    "chemin": "chemin", "impasse": "impasse", "place": "place",
    "passage": "passage", "quartier": "quartier",
}
```

**Update `name_tokens` stopwords** — add French legal suffixes as stopwords so they don't waste blocking key slots (line 75):
```python
# BEFORE:
    stop = set(_SYNONYMS.values()) | {"the", "of", "for"}

# AFTER:
    stop = set(_SYNONYMS.values()) | {"the", "of", "for", "de", "du", "des", "le", "la", "les", "et"}
```

---

## P0-E: Raise MAX_TOKEN_DF from 3000 to 10000 (1 min code change)

**Why:** With n-gram keys added, many n-grams will be common. The current MAX_TOKEN_DF=3000 will prune useful n-grams. The documentation already notes that raising to 10,000 produced identical recall for whole-word tokens, so this is safe.

**File:** `code/business_entity_resolution/src/blocking.py`

**Change line 29:**
```python
# BEFORE:
MAX_TOKEN_DF = 3000

# AFTER:
MAX_TOKEN_DF = 10000
```

---

## P0 Summary — Quick Validation After All P0 Changes

After making all P0 changes above, **validate on a small sample before committing to a full run**.

**Step 1: Quick 1% dry run to check blocking recall improvement:**
```bash
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource/code/business_entity_resolution/src
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_p0_test \
    --sample-frac 0.01
```

**What to look for in stdout:**
```
[train] blocking recall ceiling: ???/??? = X.XXXX
```
- Old value was 0.6847 (we need to get this much higher)
- Target: **>0.85** on a 1% sample. If it's still below 0.80, increase TOP_K further to 150 or 200.
- Also check `candidates/entity: mean=XX` — this will be higher (~150-250). That's expected and OK.

**Step 2: If recall is good, run 5% sample to get a validation score:**
```bash
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_p0 \
    --sample-frac 0.05
```

**What to look for:**
```
[train] global threshold=X.XX val macro F0.5=X.XXXX
[train] per-entity expected-F   val macro F0.5=X.XXXX
```
- Target: **>0.80** on val (up from 0.7028)
- If val F₀.₅ > 0.80, proceed to P1 features

**Step 3: If >0.80, go straight to full training + prediction:**
```bash
# Full training (takes ~4-8 hours with larger candidate set)
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_v2

# Predict on test set
../.venv/bin/python predict.py \
    --data-dir ../../../dataset \
    --model-dir ../model_v2 \
    --output-dir ../../../output

# Validate
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

> ⚠️ **TIME WARNING:** With TOP_K=100 + n-grams, the full training run will take longer (maybe 6-10 hours instead of 4). If time is short, use `--sample-frac 0.10` for a 10% sample which is still representative.

---

## P1-A: Add More Similarity Features (30-45 min code change)

**Why:** Our model has only 10 features. More similarity signals help the classifier distinguish true matches from hard negatives, especially for the harder cases that the expanded blocking now includes.

**File:** `code/business_entity_resolution/src/features.py`

**Add these features to `build_pair_features`:**

```python
# Add to imports at top of file:
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

# Add these inside build_pair_features(), after existing features:

    # --- NEW FEATURES ---

    # Jaro-Winkler: especially good for short strings and prefix matches
    out["name_jaro_winkler"] = [
        JaroWinkler.similarity(a, b)
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]

    # Normalized Levenshtein distance (0=identical, 1=completely different)
    out["name_levenshtein"] = [
        Levenshtein.normalized_distance(a, b)
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]

    # Address token sort ratio (handles word reordering in addresses)
    out["addr_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(pairs["addr_norm_1"], pairs["addr_norm_2"])
    ]

    # Address partial ratio (handles substring matches in addresses)
    out["addr_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(pairs["addr_norm_1"], pairs["addr_norm_2"])
    ]

    # Name token set ratio (different from token_sort — handles both reordering AND partial overlap)
    out["name_token_set_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]

    # Overlap coefficient (like Jaccard but asymmetric - good for substring business names)
    def overlap_coeff(a, b):
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / min(len(sa), len(sb))

    out["name_overlap_coeff"] = [
        overlap_coeff(a, b) for a, b in zip(name_toks_1, name_toks_2)
    ]

    # Address Jaro-Winkler
    out["addr_jaro_winkler"] = [
        JaroWinkler.similarity(a, b)
        for a, b in zip(pairs["addr_norm_1"], pairs["addr_norm_2"])
    ]

    # Character trigram Jaccard on address (catches address typos)
    out["addr_trigram_jaccard"] = [
        _jaccard(list(_trigrams(a)), list(_trigrams(b)))
        for a, b in zip(pairs["addr_norm_1"], pairs["addr_norm_2"])
    ]

    # Name length ratio (normalized) - very short vs very long name is suspicious
    out["name_len_ratio"] = [
        min(len(a), len(b)) / max(len(a), len(b)) if max(len(a), len(b)) > 0 else 1.0
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]

    # Both addresses non-empty AND matching PIN (strong combined signal)
    out["pin_and_addr_present"] = (
        (pairs["pin_1"] != "") & (pairs["pin_1"] == pairs["pin_2"]) &
        (pairs["addr_norm_1"] != "") & (pairs["addr_norm_2"] != "")
    ).astype(float)
```

**Update FEATURE_COLS** (line 14):
```python
FEATURE_COLS = [
    "name_jaccard", "name_trigram_jaccard", "name_token_sort_ratio",
    "name_partial_ratio", "name_len_diff",
    "addr_jaccard", "addr_ratio", "pin_match", "house_match",
    "addr_empty_either",
    # New features:
    "name_jaro_winkler", "name_levenshtein",
    "addr_token_sort_ratio", "addr_partial_ratio",
    "name_token_set_ratio", "name_overlap_coeff",
    "addr_jaro_winkler", "addr_trigram_jaccard",
    "name_len_ratio", "pin_and_addr_present",
]
```

**Check rapidfuzz version supports the distance module:**
```bash
../.venv/bin/python -c "from rapidfuzz.distance import Levenshtein, JaroWinkler; print('OK')"
```
If this fails, use the older API:
```python
from rapidfuzz import fuzz
# Replace JaroWinkler.similarity(a,b) with: fuzz.ratio(a, b) / 100.0  (as a fallback)
# Replace Levenshtein.normalized_distance(a,b) with: 1.0 - fuzz.ratio(a, b) / 100.0
```

---

## P1-B: Tune XGBoost Hyperparameters (5 min code change)

**File:** `code/business_entity_resolution/src/train.py`

**Change the params dict** (around line 114):
```python
# BEFORE:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": args.seed,
    }

# AFTER — deeper trees, more regularization to handle more features:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 7,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.1,
        "seed": args.seed,
    }
```

**Also increase num_boost_round** (around line 123):
```python
# BEFORE:
    booster = xgb.train(
        params, dtrain, num_boost_round=300,
        ...
    )

# AFTER:
    booster = xgb.train(
        params, dtrain, num_boost_round=600,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30, verbose_eval=50,
    )
```

---

## P1-C: Chunk Size for Larger Candidate Sets (2 min code change)

With TOP_K=100 (200 candidates/entity), the pair count grows ~3.3x. Reduce chunk sizes to avoid OOM.

**File:** `code/business_entity_resolution/src/score.py`

```python
# BEFORE:
PAIR_CHUNK_ROWS = 2_000_000

# AFTER:
PAIR_CHUNK_ROWS = 1_000_000
```

**File:** `code/business_entity_resolution/src/blocking.py`

```python
# BEFORE:
S1_CHUNK_ROWS = 25_000

# AFTER:
S1_CHUNK_ROWS = 15_000
```

---

## P2-A: Multiple Blocking Passes with Union (45-60 min, advanced)

**Why:** Instead of one blocking pass with all keys mixed into one index, run separate blocking passes with different strategies and union the candidates. This lets specialized strategies find matches that the general strategy misses, pushing recall from ~0.85 to ~0.95+. The top teams at 0.99 definitely do this.

**File:** `code/business_entity_resolution/src/blocking.py`

**Change 1: Update `block_one_source` to use multi-pass blocking.**

Modify `block_one_source` (around line 172). Instead of building one giant `full_index` and passing it to `_block_chunked` once, we will build separate indices, block them separately, and concatenate/deduplicate the results.

```python
def block_one_source(s1: pd.DataFrame, cand: pd.DataFrame, source_label: str,
                     chunk_rows: int = S1_CHUNK_ROWS) -> pd.DataFrame:
    """Return top candidates per S1 from a single candidate source (S2 or S3) via multi-pass."""
    n_by_country = cand.groupby("country", sort=False).size()
    cand_codes, cand_uniques = pd.factorize(cand["entity_id"], sort=True)
    coded = cand[["country", "name_norm", "addr_norm", "pin"]].copy()
    coded["cand_code"] = cand_codes.astype("int32")

    # 1. Build individual indices
    name_idx = _weighted_index(_token_index(coded, "cand_code", "name_norm", name_tokens), n_by_country)
    addr_idx = _weighted_index(_token_index(coded, "cand_code", "addr_norm", addr_tokens), n_by_country)
    raw_pin_idx = coded.loc[coded["pin"] != "", ["country", "pin", "cand_code"]].rename(
        columns={"pin": "token"}
    )
    pin_idx = _weighted_index(raw_pin_idx, n_by_country)
    ngram_idx = _weighted_index(_ngram_index(coded, "cand_code", "name_norm", n=4), n_by_country)

    results = []

    # PASS 1: General Pass (Name + Address + N-grams)
    full_index = pd.concat([name_idx, addr_idx, ngram_idx], ignore_index=True)
    res_general = _block_chunked(
        s1, full_index, 80, ("name", "addr"), chunk_rows, f" {source_label}-general"
    )
    results.append(res_general)
    del full_index

    # PASS 2: Address-heavy Pass (For severe name transliteration mismatch)
    res_addr = _block_chunked(
        s1, addr_idx, 30, ("addr",), chunk_rows, f" {source_label}-addr"
    )
    results.append(res_addr)

    # PASS 3: PIN-heavy Pass (For shared PIN + slight name/address overlap)
    # We mix pin and name so we don't just return everyone in the same pin code blindly,
    # but give high weight to pin.
    pin_name_idx = pd.concat([pin_idx, name_idx], ignore_index=True)
    res_pin = _block_chunked(
        s1, pin_name_idx, 30, ("pin", "name"), chunk_rows, f" {source_label}-pin"
    )
    results.append(res_pin)
    del pin_name_idx, name_idx, addr_idx, pin_idx, ngram_idx, raw_pin_idx, coded

    # Union all passes
    result = pd.concat(results, ignore_index=True)

    # Deduplicate: if same S1 and cand_code found in multiple passes, keep the max overlap
    result = result.groupby(["source1_entity_id", "cand_code"], as_index=False)["overlap"].max()

    # Map candidate codes back to real ids
    result["cand_id"] = cand_uniques.take(result["cand_code"].to_numpy())
    result = result.drop(columns=["cand_code"])

    result["source"] = source_label
    return result[["source1_entity_id", "cand_id", "source", "overlap"]]
```

> **Note:** If memory crashes occur during multi-pass blocking, reduce the `chunk_rows` or drop Pass 3.

---

## P2-B: Phonetic Blocking Keys (20 min, advanced)

**Why:** Phonetic codes map phonetically similar words (e.g. `vijan` vs `vision`) to the same key. This directly addresses the India dataset transliteration mismatches (Hindi/Bengali/Kannada → English phonetic forms) where character n-grams might fail. Top teams use phonetic blocking extensively.

**Concept:** Use `jellyfish` to generate Double Metaphone or Soundex codes for name tokens, add them as a new index layer in blocking, and add phonetic features to the model.

**Step 1: Install dependency**
```bash
../.venv/bin/pip install jellyfish
```
And add `jellyfish==1.0.0` to `code/business_entity_resolution/requirements.txt`.

**Step 2: Add phonetic token generation in `blocking.py`**
At the top of `blocking.py`, add:
```python
import jellyfish

def phonetic_tokens(df: pd.DataFrame, id_col: str, text_col: str) -> pd.DataFrame:
    """Generate Metaphone phonetic codes for name tokens."""
    def codes(text: str):
        # Generate metaphone for each token > 3 chars
        return list(dict.fromkeys(
            jellyfish.metaphone(t) for t in text.split() 
            if len(t) >= 3 and jellyfish.metaphone(t)
        ))
    
    long = pd.DataFrame({
        "country": df["country"].to_numpy(),
        id_col: df[id_col].to_numpy(),
        "token": df[text_col].map(codes).to_numpy(),
    }).explode("token")
    return long.dropna(subset=["token"])
```

**Step 3: Integrate into `_s1_long`**
In `blocking.py`, `_s1_long` function (around line 128):
```python
# Add this inside the if "name" in fields block:
    if "name" in fields:
        parts.append(_token_index(s1, "source1_entity_id", "name_norm", name_tokens))
        parts.append(_ngram_index(s1, "source1_entity_id", "name_norm", n=4))
        # Add phonetic keys:
        parts.append(phonetic_tokens(s1, "source1_entity_id", "name_norm"))
```

**Step 4: Integrate into `block_one_source`**
In `blocking.py`, `block_one_source` function:
```python
    # Build phonetic index
    phonetic_idx = _weighted_index(phonetic_tokens(coded, "cand_code", "name_norm"), n_by_country)
    
    # If using single-pass (P0-C), add phonetic_idx to full_index:
    full_index = pd.concat([name_idx, addr_idx, pin_idx, ngram_idx, phonetic_idx], ignore_index=True)
    
    # If using multi-pass (P2-A), add it to Pass 1 (general pass):
    # full_index = pd.concat([name_idx, addr_idx, ngram_idx, phonetic_idx], ignore_index=True)
```

**Step 5: Add Phonetic Features in `features.py`**
In `features.py`, add phonetic distance features to `build_pair_features`:
```python
import jellyfish

    # Add inside build_pair_features:
    out["name_phonetic_match"] = [
        int(jellyfish.metaphone(a) == jellyfish.metaphone(b))
        if a and b else 0
        for a, b in zip(pairs["name_norm_1"], pairs["name_norm_2"])
    ]
```
Add `"name_phonetic_match"` to `FEATURE_COLS`.

---

## Full Execution Sequence (Copy-Paste Commands)

### Step 1: Make ALL P0 code changes

Edit these files with the changes described above:
1. `blocking.py` — TOP_K=100, MIN_PAIR_SCORE=3.0, MAX_TOKEN_DF=10000, add `_ngram_index`, update `_s1_long` and `block_one_source`, reduce S1_CHUNK_ROWS=15000
2. `normalize.py` — French suffixes in `_SYNONYMS`, French address terms in `_ADDR_SYNONYMS`, French stopwords
3. `features.py` — Add new features and update FEATURE_COLS (P1-A)
4. `train.py` — XGBoost depth=7, eta=0.05, 600 rounds (P1-B)
5. `score.py` — PAIR_CHUNK_ROWS=1000000 (P1-C)

### Step 2: Quick 1% validation (5-10 min)

```bash
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource/code/business_entity_resolution/src
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_v2_test \
    --sample-frac 0.01
```

**Check:** blocking recall ceiling > 0.80? If yes, continue. If no, increase TOP_K to 150 or 200.

### Step 3: 5% validation (15-25 min)

```bash
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_v2_sample \
    --sample-frac 0.05
```

**Check:** val macro F₀.₅ > 0.80? If yes, proceed to full training. If not, debug which change helped and iterate.

### Step 4: Full training (4-10 hours)

```bash
../.venv/bin/python train.py \
    --data-dir ../../../dataset \
    --model-out ../model_v2
```

### Step 5: Generate predictions

```bash
../.venv/bin/python predict.py \
    --data-dir ../../../dataset \
    --model-dir ../model_v2 \
    --output-dir ../../../output
```

### Step 6: Validate

```bash
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Must see PASS.** If PASS → upload `output/matching_results.tsv` to the portal.

### Step 7: Tune selection rule (optional, 2 min)

```bash
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource/code/business_entity_resolution/src
../.venv/bin/python tune_selection.py \
    --model-dir ../model_v2 \
    --data-dir ../../../dataset
```

If a different rule beats the default, update `model_v2/model_meta.json` and re-run predict.py.

### Step 8: Re-validate + re-assemble zip

```bash
cd /Users/pushpam/Desktop/amazon_ml_challenge/student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

zip -r ../submission_v2.zip \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/src/ \
    code/business_entity_resolution/README.md \
    code/business_entity_resolution/requirements.txt \
    Documentation_template.md \
    -x "*/__pycache__/*" -x "*/.DS_Store"
```

---

## Key Data Facts the Agent Needs

| Fact | Value |
|---|---|
| Train S1 entities | 2,206,821 |
| Train S2 records | 5,034,616 |
| Train S3 records | 5,285,603 |
| Test S1 entities | 1,732,544 |
| Test S2 records | 4,887,273 |
| Test S3 records | 5,082,316 |
| **Train countries** | **US (1,323,633), India (883,188), NO France** |
| **Test countries** | **US (663,106), India (809,986), France (259,452)** |
| Ground truth match count | mean=3.67, median=4, max=11 per S1 entity |
| Singletons (no matches) | 5.58% of S1 entities |
| Ground truth total matches | 7,638,365 |
| French S2 legal suffixes | SARL (107k), SAS (75k), EURL (32k), SA (27k), SASU (25k), SCI (22k), S.A.S (6k), S.A.R.L. (5k), SÀRL (6k) |
| India non-Latin script names | ~20-30% of India S2/S3 records have Devanagari/Bengali/Tamil/Gujarati names |
| Current blocking recall | 0.6847 (training set) |
| Current val macro F₀.₅ | 0.7028 (threshold) / 0.7082 (expected-F) |
| Leaderboard score | 0.711 |
| Python venv | `.venv` at `code/business_entity_resolution/.venv/` (Python 3.14) |
| Run from | `code/business_entity_resolution/src/` |
| Python binary | `../.venv/bin/python` (from src/) |

---

## Files to Modify (Complete List)

| File | Changes | Section |
|---|---|---|
| `blocking.py` | TOP_K=100, MIN_PAIR_SCORE=3.0, MAX_TOKEN_DF=10000, S1_CHUNK_ROWS=15000, add `_ngram_index()`, update `_s1_long()`, update `block_one_source()` | P0-A/B/C/E |
| `normalize.py` | French legal suffixes in `_SYNONYMS`, French address terms in `_ADDR_SYNONYMS`, French stopwords in `name_tokens` | P0-D |
| `features.py` | Add 10 new features: Jaro-Winkler, Levenshtein, addr_token_sort_ratio, addr_partial_ratio, name_token_set_ratio, name_overlap_coeff, addr_jaro_winkler, addr_trigram_jaccard, name_len_ratio, pin_and_addr_present | P1-A |
| `train.py` | depth=7, eta=0.05, 600 rounds, early_stopping=30, min_child_weight=5, gamma=0.1 | P1-B |
| `score.py` | PAIR_CHUNK_ROWS=1000000 | P1-C |

**Do NOT modify:** `evaluate.py`, `selection.py`, `write_outputs.py`, `data_io.py`, `predict.py`, `run_pipeline.py`
(these are already correct and need no changes)

---

## What NOT to Do

- ❌ Do NOT hard-code `{US, India, France}` — use country as a generic string partition
- ❌ Do NOT use any external API, database, or web lookup — IMMEDIATE DISQUALIFICATION
- ❌ Do NOT use `model_dryrun/` for anything — it's a 1% sample, F₀.₅=0.459
- ❌ Do NOT include `.venv/`, `model/`, `dataset/`, or `__pycache__/` in the submission zip
- ❌ Do NOT upload `candidate_pairs.tsv` to the leaderboard — only `matching_results.tsv`
- ❌ Do NOT skip the validator before uploading

---

## Expected Score Progression

| After | Expected Val F₀.₅ | Expected Leaderboard | Why |
|---|---|---|---|
| **Baseline (current)** | 0.7028 | 0.711 | — |
| **P0-A+B** (TOP_K=100 + lower MIN_PAIR_SCORE) | ~0.77-0.82 | ~0.78-0.83 | Blocking recall jumps from 0.68 to ~0.80+ |
| **+P0-C** (n-gram keys) | ~0.82-0.87 | ~0.83-0.88 | Catches typos and transliterations |
| **+P0-D** (French suffixes) | ~0.83-0.88 | ~0.84-0.89 | Fixes 15% of test set |
| **+P1-A** (more features) | ~0.85-0.92 | ~0.86-0.93 | Better classifier separation |
| **+P1-B** (tuned XGBoost) | ~0.86-0.93 | ~0.87-0.94 | Deeper model exploits more features |
| **+P2** (phonetic + multi-pass) | ~0.88-0.95 | ~0.89-0.96 | Catches the hardest transliteration cases |

> These are rough estimates. The actual scores depend on how much blocking recall improves, which is the dominant factor. The P0 changes are the highest-leverage and should be done first.

---

*Generated: 2026-09-26 21:53 IST — For execution by a code agent on the amazon_ml_challenge workspace*
