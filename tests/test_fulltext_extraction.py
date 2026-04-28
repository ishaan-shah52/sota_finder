from src.rule_extract import enrich_records
from src.schemas import PaperRecord


def test_methods_text_extracts_cv_and_preprocessing():
    paper = PaperRecord(
        title="Wearable sleep staging on DREAMT",
        notes=(
            "FullText: We evaluated the model using five-fold subject-wise "
            "cross-validation. During preprocessing, BVP signals were filtered "
            "with a 3-11 Hz bandpass, normalized, downsampled to 32 Hz, and "
            "segmented into 5-second windows."
        ),
    )

    enriched = enrich_records([paper])[0]

    assert enriched.cross_validation == "5-fold subject-wise"
    assert "bandpass 3–11 Hz" in enriched.preprocessing_steps
    assert "sampling rate 32 Hz" in enriched.preprocessing_steps
    assert "5-second window" in enriched.preprocessing_steps
    assert "normalization" in enriched.preprocessing_steps
