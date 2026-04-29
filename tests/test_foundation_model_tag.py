from src.rule_extract import enrich_records
from src.schemas import PaperRecord


def test_foundation_model_tag_detects_explicit_foundation_model():
    papers = enrich_records([
        PaperRecord(
            title="FEMBA: Efficient EEG Analysis with a Foundation Model",
            notes=(
                "Abstract: FEMBA is a self-supervised EEG foundation model "
                "fine-tuned on downstream artifact detection tasks."
            ),
        )
    ])

    assert papers[0].foundation_model == "yes"


def test_foundation_model_tag_marks_no_when_no_signal_found():
    papers = enrich_records([
        PaperRecord(
            title="A CNN for sleep stage classification",
            notes="Abstract: A convolutional network is evaluated on a sleep dataset.",
        )
    ])

    assert papers[0].foundation_model == "no"
