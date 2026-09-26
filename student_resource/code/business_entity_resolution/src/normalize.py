"""Text normalization for business_name / business_address columns.

Vectorized over pandas Series (no per-row python loops) since files run
5-10M rows.
"""
import re
import pandas as pd
from unidecode import unidecode

# pandas 3's default Arrow-backed "str" dtype makes the python-level string
# ops used throughout this pipeline (.map, zip, set-building over millions of
# rows) 10-100x slower -- every element access pays a PyArrow-scalar
# conversion. Force the classic numpy-object string dtype everywhere.
pd.set_option("future.infer_string", False)

# legal-suffix / common-word synonyms folded to one canonical token
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

# address abbreviation -> canonical token
_ADDR_SYNONYMS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "apartment": "apt", "unit": "unit", "suite": "ste",
    # French address terms
    "rue": "rue", "allée": "allee", "allee": "allee",
    "chemin": "chemin", "impasse": "impasse", "place": "place",
    "passage": "passage", "quartier": "quartier",
}

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_PIN_RE = re.compile(r"(\d{5,6})(?!.*\d{5,6})")  # last 5-6 digit run
_LEADING_NUM_RE = re.compile(r"^(\d+)")


def _clean_ascii(series: pd.Series) -> pd.Series:
    """lowercase, transliterate to ascii, strip punctuation, collapse whitespace."""
    s = series.fillna("").map(unidecode).str.lower()
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _apply_synonyms(series: pd.Series, mapping: dict) -> pd.Series:
    def sub(text: str) -> str:
        return " ".join(mapping.get(tok, tok) for tok in text.split())
    return series.map(sub)


def normalize_names(series: pd.Series) -> pd.Series:
    return _apply_synonyms(_clean_ascii(series), _SYNONYMS)


def normalize_addresses(series: pd.Series) -> pd.Series:
    return _apply_synonyms(_clean_ascii(series), _ADDR_SYNONYMS)


def extract_pin(addr_raw: pd.Series) -> pd.Series:
    """Last 5-6 digit run in the raw address, or '' if none."""
    return addr_raw.fillna("").str.extract(_PIN_RE, expand=False).fillna("")


def extract_house_number(addr_norm: pd.Series) -> pd.Series:
    """Leading digit run of the normalized address, or '' if none."""
    return addr_norm.str.extract(_LEADING_NUM_RE, expand=False).fillna("")


def name_tokens(name_norm: pd.Series, min_len: int = 3) -> pd.Series:
    """Distinct significant tokens per row, dropping short/legal-suffix noise words.

    Each token appears at most once per record: a repeated word is not extra evidence,
    and de-duplicating here is free, whereas de-duplicating the exploded index later
    costs a hash over tens of millions of rows.
    """
    stop = set(_SYNONYMS.values()) | {"the", "of", "for", "de", "du", "des", "le", "la", "les", "et"}

    def toks(text: str):
        return list(dict.fromkeys(
            t for t in text.split() if len(t) >= min_len and t not in stop
        ))

    return name_norm.map(toks)


def addr_tokens(addr_norm: pd.Series, min_len: int = 3) -> pd.Series:
    """Distinct significant address tokens per row (see name_tokens on de-duplication)."""
    stop = set(_ADDR_SYNONYMS.values())

    def toks(text: str):
        return list(dict.fromkeys(
            t for t in text.split()
            if len(t) >= min_len and t not in stop and not t.isdigit()
        ))

    return addr_norm.map(toks)


def prep_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized columns for blocking/features, keeping only what is needed.

    Deliberately compact: the raw name/address columns are dropped once normalized,
    and token lists are NOT materialised -- they are derived on demand by
    name_tokens()/addr_tokens(). Storing them costs ~1GB per source (a separate
    python list object per row) and three sources then exceed available memory.
    """
    out = pd.DataFrame({"entity_id": df["entity_id"], "country": df["country"]})
    out["name_norm"] = normalize_names(df["business_name"])
    out["addr_norm"] = normalize_addresses(df["business_address"])
    out["pin"] = extract_pin(df["business_address"])
    out["house_no"] = extract_house_number(out["addr_norm"])
    return out


def demo():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S2-1", "S3-1"],
        "business_name": ["Iris Brothers Pvt Ltd", "IRIS BROTHERS PRIVATE LIMITED", "wilfordhancock.com"],
        "business_address": ["123 Main St, Springfield, IL 62701", "123 MAIN STREET, SPRINGFIELD, IL", ""],
        "country": ["India", "India", "US"],
    })
    out = prep_source(df)
    assert out["name_norm"][0] == out["name_norm"][1] == "iris brothers pvt ltd"
    assert out["pin"][0] == "62701" and out["pin"][1] == ""
    assert out["house_no"][0] == "123" == out["house_no"][1]
    assert "business_name" not in out.columns, "raw columns must be dropped"
    toks = name_tokens(out["name_norm"])
    assert "iris" in toks[0] and "brothers" in toks[0]
    assert "ltd" not in toks[0] and "pvt" not in toks[0], "legal suffixes are stopwords"
    print("normalize.py demo OK")


if __name__ == "__main__":
    demo()
