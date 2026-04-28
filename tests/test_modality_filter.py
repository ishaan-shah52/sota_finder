from src.extract import _has_target_dataset_evidence
from src.schemas import PaperRecord


def test_modality_filter_preserves_dataset_page_sota_without_selected_modality():
    paper = PaperRecord(
        title=(
            "Addressing Wearable Sleep Tracking Inequity: A New Dataset and "
            "Novel Methods for a Population with Sleep Disorders"
        ),
        datasets_used=["DREAMT"],
        notes="Auto-extracted from PhysioNet dataset page.",
    )

    assert _has_target_dataset_evidence(
        paper,
        "DREAMT",
        "Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology",
    )


def test_modality_filter_does_not_preserve_unrelated_ecg_paper():
    paper = PaperRecord(
        title="ECG-Lens: Benchmarking ML and DL Models on PTB-XL Dataset",
        datasets_used=["DREAMT"],
        notes="Auto-extracted from arXiv benchmark/model expansion.",
    )

    assert not _has_target_dataset_evidence(
        paper,
        "DREAMT",
        "Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology",
    )
