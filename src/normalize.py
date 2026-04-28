"""Lightweight normalization helpers — no mutation of original records."""

from __future__ import annotations

METRIC_ALIASES: dict[str, str] = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "f1": "f1",
    "f1-score": "f1",
    "f1 score": "f1",
    "macro f1": "f1-macro",
    "macro-f1": "f1-macro",
    "weighted f1": "f1-weighted",
    "weighted-f1": "f1-weighted",
    "cohen's kappa": "kappa",
    "cohens kappa": "kappa",
    "kappa": "kappa",
    "mf1": "f1-macro",
    "mcc": "mcc",
    "auc": "auc",
    "auroc": "auc",
    "roc-auc": "auc",
}

TASK_ALIASES: dict[str, str] = {
    "sleep staging": "sleep-staging",
    "sleep stage classification": "sleep-staging",
    "sleep classification": "sleep-staging",
    "ecg classification": "ecg-classification",
    "arrhythmia classification": "ecg-arrhythmia",
    "arrhythmia detection": "ecg-arrhythmia",
    "seizure detection": "eeg-seizure-detection",
    "seizure classification": "eeg-seizure-detection",
    "eeg classification": "eeg-classification",
    "emotion recognition": "eeg-emotion-recognition",
}


def normalize_metric(raw: str) -> str:
    return METRIC_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def normalize_task(raw: str) -> str:
    return TASK_ALIASES.get(raw.strip().lower(), raw.strip().lower())
