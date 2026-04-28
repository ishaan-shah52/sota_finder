from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, field_validator


UNKNOWN = "UNKNOWN"


class PaperRecord(BaseModel):
    title: str
    year: int | str = UNKNOWN
    paper_url: str = UNKNOWN

    # Dataset context
    datasets_used: List[str] = []
    target_dataset_only: bool | str = UNKNOWN

    # Task definition
    task: str = UNKNOWN
    label_granularity: str = UNKNOWN  # e.g. "5-class (W/N1/N2/N3/REM)"

    # Split strategy
    train_split: str = UNKNOWN
    val_split: str = UNKNOWN
    test_split: str = UNKNOWN
    cross_validation: str = UNKNOWN  # e.g. "10-fold subject-wise" or UNKNOWN
    split_type: str = UNKNOWN        # "subject-wise", "random-window", "mixed", UNKNOWN

    # Signal processing
    preprocessing_steps: List[str] = []
    modalities: List[str] = []

    # Model
    model_name: str = UNKNOWN
    models_compared: List[str] = []  # baselines listed in the paper

    # Results
    metric_name: str = UNKNOWN
    metric_value: float | str = UNKNOWN
    result_scope: str = UNKNOWN      # "test", "val", "unknown"

    # Reviewer metadata
    notes: str = ""
    confidence: str = UNKNOWN        # "high", "medium", "low"
    source_snippet: str = ""         # quoted text from paper supporting the numbers

    # Quality assessment (populated by src/red_flags.py)
    # sota_category: "STRONG SOTA CANDIDATE" | "WEAK SOTA CANDIDATE" | "RELATED WORK ONLY" | UNKNOWN
    sota_category: str = UNKNOWN
    # confidence_score: 0.0–1.0 after applying code-availability and venue-tier multipliers
    confidence_score: float | str = UNKNOWN
    # Individual flag strings: "CRITICAL|CHECK_N|human message" or "SECONDARY|CHECK_N|human message"
    red_flags: List[str] = []

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        allowed = {"high", "medium", "low", UNKNOWN}
        if v not in allowed:
            raise ValueError(f"confidence must be one of {allowed}, got {v!r}")
        return v

    @field_validator("result_scope")
    @classmethod
    def validate_result_scope(cls, v: str) -> str:
        allowed = {"test", "val", "unknown", UNKNOWN}
        if v not in allowed:
            raise ValueError(f"result_scope must be one of {allowed}, got {v!r}")
        return v

    @field_validator("split_type")
    @classmethod
    def validate_split_type(cls, v: str) -> str:
        allowed = {"subject-wise", "random-window", "mixed", UNKNOWN}
        if v not in allowed:
            raise ValueError(f"split_type must be one of {allowed}, got {v!r}")
        return v
