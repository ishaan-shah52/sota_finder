from src.red_flags import apply_red_flags
from src.rule_extract import dataset_use_evidence, top_n
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


def test_isruc_sleep_curated_seed_ranks_above_unverified_candidates():
    curated = _curated_sota_records("ISRUC-SLEEP")[0]
    unverified = PaperRecord(
        title="Generic sleep stage classifier with strong reported accuracy",
        year=2026,
        datasets_used=["ISRUC-SLEEP"],
        task="sleep staging",
        label_granularity="5-class",
        split_type="subject-wise",
        cross_validation="10-fold subject-wise",
        modalities=["PSG"],
        model_name="GenericNet",
        metric_name="accuracy",
        metric_value=99.0,
        result_scope="test",
        confidence="high",
        notes="Auto-extracted from arXiv benchmark/model expansion. Abstract: high accuracy on sleep data.",
    )

    assert dataset_use_evidence(curated) == "direct metadata mention"
    assert top_n([unverified, curated], n=2)[0].title == curated.title


def test_isruc_sleep_curated_seed_stays_near_top_after_red_flags():
    curated = _curated_sota_records("ISRUC-S3")[0]
    competitor = PaperRecord(
        title="Another ISRUC-S3 sleep staging benchmark",
        year=2026,
        paper_url="https://example.com/isruc-s3-benchmark",
        datasets_used=["ISRUC-S3"],
        task="sleep staging",
        label_granularity="5-class",
        split_type="subject-wise",
        cross_validation="10-fold subject-wise",
        modalities=["PSG"],
        model_name="BenchmarkNet",
        models_compared=["previous SOTA"],
        metric_name="accuracy",
        metric_value=85.0,
        result_scope="test",
        confidence="high",
        notes=(
            "Abstract: evaluated on ISRUC-S3 with subject-independent 10-fold "
            "cross-validation and compared against previous SOTA baselines."
        ),
    )

    ranked = top_n(apply_red_flags([competitor, curated]), n=2)

    assert ranked[0].title == curated.title
