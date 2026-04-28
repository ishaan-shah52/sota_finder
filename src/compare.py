"""Group papers into comparable / partially comparable / not comparable buckets.

Grouping rules (ALL must match for "comparable"):
  - same normalized task
  - same normalized metric_name
  - same label_granularity
  - same split_type (and neither is UNKNOWN)

Papers are "partially comparable" when they share task + metric but differ on
label_granularity or split_type, or when one value is UNKNOWN.

Everything else is "not comparable."

Additional flags (do not block grouping, but are recorded):
  - uses multiple datasets
  - result_scope is not "test"
  - confidence is "low"
  - metric value is UNKNOWN
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.schemas import PaperRecord, UNKNOWN
from src.normalize import normalize_metric, normalize_task


@dataclass
class ComparisonGroup:
    key: Tuple[str, str, str, str]   # (task, metric, label_granularity, split_type)
    papers: List[PaperRecord] = field(default_factory=list)

    @property
    def task(self) -> str:
        return self.key[0]

    @property
    def metric(self) -> str:
        return self.key[1]

    @property
    def label_granularity(self) -> str:
        return self.key[2]

    @property
    def split_type(self) -> str:
        return self.key[3]


@dataclass
class ComparisonResult:
    comparable: List[ComparisonGroup]
    partially_comparable: List[ComparisonGroup]
    not_comparable: List[PaperRecord]


def _caveats(paper: PaperRecord) -> List[str]:
    """Return a list of caveat strings for a paper."""
    caveats: List[str] = []
    if len(paper.datasets_used) > 1:
        caveats.append(f"Uses multiple datasets: {', '.join(paper.datasets_used)}")
    if paper.target_dataset_only is False:
        caveats.append("Results include data outside the target dataset")
    if paper.result_scope not in ("test", UNKNOWN):
        caveats.append(f"Result scope: {paper.result_scope} (not a held-out test set)")
    if paper.result_scope == UNKNOWN:
        caveats.append("Result scope unknown — may be validation or train performance")
    if paper.confidence == "low":
        caveats.append("Low reviewer confidence in extracted numbers")
    if paper.metric_value == UNKNOWN:
        caveats.append("Metric value not available")
    if paper.split_type == UNKNOWN:
        caveats.append("Split type unknown — comparability uncertain")
    if paper.cross_validation == UNKNOWN and paper.split_type == UNKNOWN:
        caveats.append("No CV / split info reported")
    return caveats


def _group_key(paper: PaperRecord) -> Tuple[str, str, str, str]:
    return (
        normalize_task(paper.task) if paper.task != UNKNOWN else UNKNOWN,
        normalize_metric(paper.metric_name) if paper.metric_name != UNKNOWN else UNKNOWN,
        paper.label_granularity,
        paper.split_type,
    )


def _is_comparable_key(key: Tuple[str, str, str, str]) -> bool:
    task, metric, granularity, split = key
    if UNKNOWN in (task, metric, granularity, split):
        return False
    return True


def _is_partial_key(key: Tuple[str, str, str, str]) -> bool:
    task, metric, granularity, split = key
    # Must at least agree on task and metric
    if task == UNKNOWN or metric == UNKNOWN:
        return False
    return True


def group_papers(papers: List[PaperRecord]) -> ComparisonResult:
    full_groups: Dict[Tuple, List[PaperRecord]] = defaultdict(list)
    partial_groups: Dict[Tuple, List[PaperRecord]] = defaultdict(list)
    not_comparable: List[PaperRecord] = []

    for paper in papers:
        key = _group_key(paper)
        if _is_comparable_key(key):
            full_groups[key].append(paper)
        elif _is_partial_key(key):
            # Use (task, metric) as partial key so papers with same task+metric cluster
            partial_key = (key[0], key[1], key[2], key[3])  # keep full key for display
            partial_groups[partial_key].append(paper)
        else:
            not_comparable.append(paper)

    comparable = [
        ComparisonGroup(key=k, papers=v)
        for k, v in full_groups.items()
        if len(v) >= 1  # include singletons so they appear in the report
    ]
    partially_comparable = [
        ComparisonGroup(key=k, papers=v)
        for k, v in partial_groups.items()
    ]

    return ComparisonResult(
        comparable=comparable,
        partially_comparable=partially_comparable,
        not_comparable=not_comparable,
    )


def get_caveats(paper: PaperRecord) -> List[str]:
    return _caveats(paper)
