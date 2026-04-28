"""Tests for normalization helpers."""

from src.normalize import normalize_metric, normalize_task


def test_metric_aliases():
    assert normalize_metric("acc") == "accuracy"
    assert normalize_metric("Accuracy") == "accuracy"
    assert normalize_metric("F1-Score") == "f1"
    assert normalize_metric("macro f1") == "f1-macro"
    assert normalize_metric("Cohen's Kappa") == "kappa"
    assert normalize_metric("AUROC") == "auc"


def test_metric_passthrough():
    assert normalize_metric("specificity") == "specificity"


def test_task_aliases():
    assert normalize_task("Sleep Staging") == "sleep-staging"
    assert normalize_task("sleep stage classification") == "sleep-staging"
    assert normalize_task("Arrhythmia Detection") == "ecg-arrhythmia"
    assert normalize_task("seizure detection") == "eeg-seizure-detection"


def test_task_passthrough():
    assert normalize_task("custom-task-xyz") == "custom-task-xyz"
