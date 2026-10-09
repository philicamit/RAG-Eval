from pathlib import Path

import pytest

from rag_eval.dataset import load_golden_dataset


DATASET = Path(__file__).resolve().parent.parent / "rag_eval" / "datasets" / "got_golden.jsonl"


def test_golden_dataset_loads_and_has_required_fields():
    cases = load_golden_dataset(DATASET)
    assert len(cases) >= 10
    answerable = [case for case in cases if not case.unanswerable]
    unanswerable = [case for case in cases if case.unanswerable]
    assert answerable
    assert unanswerable
    for case in answerable:
        assert case.input
        assert case.expected_output
        assert case.expected_keywords or case.expected_context


def test_max_cases_is_respected():
    cases = load_golden_dataset(DATASET, max_cases=3)
    assert len(cases) == 3


def test_missing_dataset_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_golden_dataset(tmp_path / "missing.jsonl")
