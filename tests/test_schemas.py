"""Tests for schema validation."""

import pytest
from pydantic import ValidationError
from src.schemas import PaperRecord, UNKNOWN


def test_default_unknown():
    p = PaperRecord(title="Test Paper")
    assert p.task == UNKNOWN
    assert p.metric_name == UNKNOWN
    assert p.year == UNKNOWN


def test_valid_confidence():
    p = PaperRecord(title="Test", confidence="high")
    assert p.confidence == "high"


def test_invalid_confidence():
    with pytest.raises(ValidationError):
        PaperRecord(title="Test", confidence="very_high")


def test_valid_result_scope():
    for scope in ("test", "val", "unknown", UNKNOWN):
        p = PaperRecord(title="Test", result_scope=scope)
        assert p.result_scope == scope


def test_invalid_result_scope():
    with pytest.raises(ValidationError):
        PaperRecord(title="Test", result_scope="train")


def test_valid_split_type():
    for st in ("subject-wise", "random-window", "mixed", UNKNOWN):
        p = PaperRecord(title="Test", split_type=st)
        assert p.split_type == st


def test_invalid_split_type():
    with pytest.raises(ValidationError):
        PaperRecord(title="Test", split_type="stratified-random")


def test_list_fields_default_empty():
    p = PaperRecord(title="Test")
    assert p.datasets_used == []
    assert p.preprocessing_steps == []
    assert p.modalities == []
    assert p.models_compared == []
