"""Tests for grouping logic."""

from src.schemas import PaperRecord, UNKNOWN
from src.compare import group_papers, get_caveats
from src.mock_data import MOCK_PAPERS


def _make(task, metric, granularity, split_type, **kwargs):
    return PaperRecord(
        title=f"{task}-{metric}-{split_type}",
        task=task,
        metric_name=metric,
        label_granularity=granularity,
        split_type=split_type,
        **kwargs,
    )


def test_two_identical_papers_grouped():
    p1 = _make("Sleep Staging", "F1-macro", "5-class", "subject-wise", metric_value=0.80)
    p2 = _make("Sleep Staging", "F1-macro", "5-class", "subject-wise", metric_value=0.78)
    result = group_papers([p1, p2])
    assert len(result.comparable) == 1
    assert len(result.comparable[0].papers) == 2


def test_different_split_types_not_comparable():
    p1 = _make("Sleep Staging", "Accuracy", "5-class", "subject-wise")
    p2 = _make("Sleep Staging", "Accuracy", "5-class", "random-window")
    result = group_papers([p1, p2])
    # Different split types → different groups, not merged
    assert all(len(g.papers) == 1 for g in result.comparable)


def test_different_metrics_not_grouped_together():
    p1 = _make("Sleep Staging", "F1-macro", "5-class", "subject-wise")
    p2 = _make("Sleep Staging", "Accuracy", "5-class", "subject-wise")
    result = group_papers([p1, p2])
    keys = [g.key for g in result.comparable]
    # They should be in separate groups
    assert len(keys) == 2


def test_unknown_task_goes_to_not_comparable():
    p = _make(UNKNOWN, "F1-macro", "5-class", "subject-wise")
    result = group_papers([p])
    assert len(result.not_comparable) == 1


def test_unknown_metric_goes_to_not_comparable():
    p = _make("Sleep Staging", UNKNOWN, "5-class", "subject-wise")
    result = group_papers([p])
    assert len(result.not_comparable) == 1


def test_unknown_split_type_goes_to_partial():
    p = _make("Sleep Staging", "F1-macro", "5-class", UNKNOWN)
    result = group_papers([p])
    assert len(result.partially_comparable) == 1


def test_mock_data_runs_without_error():
    result = group_papers(MOCK_PAPERS)
    total = (
        sum(len(g.papers) for g in result.comparable)
        + sum(len(g.papers) for g in result.partially_comparable)
        + len(result.not_comparable)
    )
    assert total == len(MOCK_PAPERS)


def test_caveats_multi_dataset():
    p = PaperRecord(
        title="Multi",
        datasets_used=["SHHS", "Sleep-EDF"],
        result_scope="test",
        confidence="high",
    )
    caveats = get_caveats(p)
    assert any("multiple datasets" in c for c in caveats)


def test_caveats_low_confidence():
    p = PaperRecord(title="Low", confidence="low", result_scope="test")
    caveats = get_caveats(p)
    assert any("Low reviewer confidence" in c for c in caveats)


def test_caveats_unknown_result_scope():
    p = PaperRecord(title="Unknown scope", result_scope=UNKNOWN)
    caveats = get_caveats(p)
    assert any("Result scope unknown" in c for c in caveats)
