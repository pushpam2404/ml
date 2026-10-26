import importlib

import numpy as np
import pandas as pd
import pytest

from evaluate import f_beta, macro_f_beta
from normalize import normalize_names, normalize_addresses, extract_pin, name_tokens
from selection import select_by_threshold, select_expected_f


@pytest.mark.parametrize("module", ["normalize", "evaluate", "selection", "features",
                                    "write_outputs", "score", "blocking"])
def test_module_self_check(module):
    importlib.import_module(module).demo()


def test_f_beta_matches_problem_statement_example():
    # precision 2/3, recall 1.0 -> 0.714 in the challenge README
    assert f_beta(2 / 3, 1.0) == pytest.approx(0.7143, abs=1e-3)


def test_macro_f_beta_scores_singletons():
    truth = {"S1-1": set(), "S1-2": set(), "S1-3": {"S2-1"}}
    pred = {"S1-1": set(), "S1-2": {"S2-9"}, "S1-3": {"S2-1"}}
    # correct empty = 1.0, any match on a singleton = 0.0, exact match = 1.0
    assert macro_f_beta(pred, truth) == pytest.approx(2 / 3)


def test_missing_prediction_counts_as_empty():
    assert macro_f_beta({}, {"S1-1": {"S2-1"}}) == 0.0
    assert macro_f_beta({}, {"S1-1": set()}) == 1.0


def test_normalize_folds_legal_suffixes_and_accents():
    out = normalize_names(pd.Series(["Iris Brothers Private Limited", "Société Dupont Fils SARL"]))
    assert out[0] == "iris brothers pvt ltd"
    assert out[1] == "soc dupont fils sarl"  # unidecode first, then synonym folding


def test_france_is_not_special_cased():
    # the test set has a country absent from training; nothing may key on a fixed list
    import blocking
    import inspect
    # the module self-check legitimately names countries, so look at the real code paths
    code = "".join(inspect.getsource(f) for f in (
        blocking.generate_candidates, blocking.block_one_source, blocking._weighted_index))
    for country in ("France", "India", "US"):
        assert country not in code


def test_extract_pin_takes_last_digit_run():
    pins = extract_pin(pd.Series(["12 Main St, Springfield IL 62701", "no digits here", "Flat 400001, Mumbai 400002"]))
    assert list(pins) == ["62701", "", "400002"]


def test_name_tokens_drop_legal_suffixes_and_dedupe():
    toks = name_tokens(pd.Series(["iris iris brothers pvt ltd"]))[0]
    assert toks == ["iris", "brothers"]


def test_select_by_threshold_is_inclusive():
    pairs = pd.DataFrame({"source1_entity_id": ["a", "a"], "cand_id": ["x", "y"], "prob": [0.5, 0.49]})
    assert list(select_by_threshold(pairs, 0.5)["cand_id"]) == ["x"]


def test_expected_f_prefers_empty_when_nothing_is_likely():
    pairs = pd.DataFrame({"source1_entity_id": ["a"] * 3, "cand_id": ["x", "y", "z"],
                          "prob": [0.05, 0.04, 0.03]})
    assert select_expected_f(pairs).empty


def test_expected_f_keeps_confident_prefix_only():
    pairs = pd.DataFrame({"source1_entity_id": ["a"] * 4, "cand_id": list("wxyz"),
                          "prob": [0.98, 0.95, 0.4, 0.03]})
    kept = set(select_expected_f(pairs)["cand_id"])
    assert {"w", "x"} <= kept and "z" not in kept
