"""Train, predict and validate end to end on made-up data.

This checks that the pipeline runs and produces a valid submission, including for a
country the training data never had. It says nothing about accuracy on the real
challenge data, which is not in this repository.
"""
import json
import os
import subprocess
import sys

import pytest

import synthetic

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
VALIDATOR = os.path.join(HERE, "..", "..", "..", "utils", "validate_submission.py")


def run(*args):
    return subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    data, model, out = (str(root / n) for n in ("dataset", "model", "output"))
    synthetic.make_dataset(data, n_train=1500, n_test=500)
    run(os.path.join(SRC, "run_pipeline.py"), "--data-dir", data, "--model-out", model, "--output-dir", out)
    return data, model, out


def test_validator_passes(pipeline):
    data, _, out = pipeline
    text = run(VALIDATOR, "--matching", os.path.join(out, "matching_results.tsv"),
               "--candidate", os.path.join(out, "candidate_pairs.tsv"),
               "--test-dir", os.path.join(data, "test"), "--check-ids")
    assert "PASS" in text


def test_every_test_entity_has_a_row_including_unseen_country(pipeline):
    data, _, out = pipeline
    with open(os.path.join(data, "test", "test_source1.tsv"), encoding="utf-8") as f:
        rows = [l.split("\t") for l in f.read().splitlines()[1:]]
    expected_ids = {r[0] for r in rows}
    assert any(r[3] == "France" for r in rows)
    with open(os.path.join(out, "matching_results.tsv"), encoding="utf-8") as f:
        got_ids = [l.split("\t")[0] for l in f.read().splitlines()[1:]]
    assert set(got_ids) == expected_ids and len(got_ids) == len(expected_ids)


def test_matches_are_subset_of_candidates(pipeline):
    _, _, out = pipeline

    def load(name):
        with open(os.path.join(out, name), encoding="utf-8") as f:
            return {l.split("\t")[0]: set(filter(None, l.rstrip("\n").split("\t")[1].split(",")))
                    for l in f.read().splitlines()[1:]}

    matches, cands = load("matching_results.tsv"), load("candidate_pairs.tsv")
    assert all(matches[k] <= cands[k] for k in matches)


def test_model_meta_is_sane(pipeline):
    _, model, _ = pipeline
    with open(os.path.join(model, "model_meta.json")) as f:
        meta = json.load(f)
    assert meta["blocking_recall_ceiling"] > 0.9
    assert meta["val_macro_f0_5"] > 0.8
