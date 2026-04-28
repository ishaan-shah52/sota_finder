from src.rule_extract import top_n
from src.schemas import PaperRecord
from src.extract import _curated_sota_records


def test_final_ranking_prefers_higher_confidence_with_same_dataset_evidence():
    low_conf = PaperRecord(
        title="DREAMT lower confidence",
        datasets_used=["DREAMT"],
        notes="Abstract: evaluated on DREAMT.",
        sota_category="STRONG SOTA CANDIDATE",
        confidence_score=0.6,
    )
    high_conf = low_conf.model_copy(update={
        "title": "DREAMT higher confidence",
        "confidence_score": 0.95,
    })

    assert top_n([low_conf, high_conf], n=2)[0].title == "DREAMT higher confidence"


def test_final_ranking_keeps_dataset_evidence_above_unverified_high_confidence():
    verified = PaperRecord(
        title="DREAMT verified",
        datasets_used=["DREAMT"],
        notes="Auto-extracted from PhysioNet dataset page.",
        sota_category="WEAK SOTA CANDIDATE",
        confidence_score=0.6,
    )
    unverified = PaperRecord(
        title="Generic high confidence",
        datasets_used=["DREAMT"],
        notes="Auto-extracted from arXiv benchmark/model expansion.",
        sota_category="STRONG SOTA CANDIDATE",
        confidence_score=0.95,
    )

    assert top_n([unverified, verified], n=2)[0].title == "DREAMT verified"


def test_final_ranking_uses_confidence_before_category_for_same_dataset_evidence():
    strong_lower_conf = PaperRecord(
        title="TUAR strong lower confidence",
        datasets_used=["TUAR"],
        notes="Abstract: evaluated on TUAR.",
        sota_category="STRONG SOTA CANDIDATE",
        confidence_score=0.6,
    )
    weak_higher_conf = strong_lower_conf.model_copy(update={
        "title": "TUAR weak higher confidence",
        "sota_category": "WEAK SOTA CANDIDATE",
        "confidence_score": 0.95,
    })

    assert top_n([strong_lower_conf, weak_higher_conf], n=2)[0].title == "TUAR weak higher confidence"


def test_isruc_sleep_curated_seed_is_available_for_common_aliases():
    for dataset_name in ["ISRUC", "ISRUC-Sleep", "ISRUC Sleep Dataset", "ISRUC-S1", "ISRUC-S3"]:
        records = _curated_sota_records(dataset_name)
        assert records
        assert records[0].title == "Towards interpretable sleep stage classification with a multi-stream fusion network"
