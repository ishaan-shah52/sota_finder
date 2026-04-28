"""Tests for retrieval matching heuristics."""

from src.fetch_papers import _has_dataset_name_signal, _looks_like_dataset_record


def test_common_word_acronym_needs_dataset_context():
    assert not _has_dataset_name_signal("the problem was faced by many patients", "FACED")
    assert _has_dataset_name_signal("FACED is an EEG dataset for emotion recognition", "FACED")
    assert _has_dataset_name_signal("faced affective EEG dataset", "FACED")


def test_repository_records_must_look_like_datasets():
    assert not _looks_like_dataset_record(
        "Egyptian Faced Vases",
        "A museum object record with no biomedical signals.",
        "FACED",
    )
    assert _looks_like_dataset_record(
        "FACED dataset",
        "A finer-grained affective EEG dataset with emotion labels.",
        "FACED",
    )
