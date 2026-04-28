"""Render CSV and Markdown reports from comparison results."""

from __future__ import annotations

import csv
import os
from typing import List

from src.compare import ComparisonResult, ComparisonGroup, get_caveats
from src.schemas import PaperRecord, UNKNOWN


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "title", "year", "paper_url",
    "datasets_used", "target_dataset_only",
    "task", "label_granularity",
    "train_split", "val_split", "test_split",
    "cross_validation", "split_type",
    "preprocessing_steps", "modalities",
    "model_name", "models_compared",
    "metric_name", "metric_value", "result_scope",
    "notes", "confidence", "source_snippet",
    "sota_category", "confidence_score", "red_flags",
]


def _paper_to_row(paper: PaperRecord) -> dict:
    row = {}
    for f in CSV_FIELDS:
        val = getattr(paper, f)
        if isinstance(val, list):
            val = "; ".join(val) if val else ""
        elif val is True:
            val = "yes"
        elif val is False:
            val = "no"
        row[f] = val
    return row


def write_csv(papers: List[PaperRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for paper in papers:
            writer.writerow(_paper_to_row(paper))


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _paper_block(paper: PaperRecord, idx: int) -> str:
    caveats = get_caveats(paper)
    url = paper.paper_url if paper.paper_url != UNKNOWN else None
    title_md = f"[{paper.title}]({url})" if url else paper.title

    datasets = ", ".join(paper.datasets_used) if paper.datasets_used else UNKNOWN
    preprocessing = (
        "; ".join(paper.preprocessing_steps) if paper.preprocessing_steps else UNKNOWN
    )
    modalities = ", ".join(paper.modalities) if paper.modalities else UNKNOWN
    baselines = (
        ", ".join(paper.models_compared) if paper.models_compared else UNKNOWN
    )

    split_info = []
    if paper.train_split != UNKNOWN:
        split_info.append(f"Train: {paper.train_split}")
    if paper.val_split != UNKNOWN:
        split_info.append(f"Val: {paper.val_split}")
    if paper.test_split != UNKNOWN:
        split_info.append(f"Test: {paper.test_split}")
    split_str = " | ".join(split_info) if split_info else UNKNOWN

    metric_str = (
        f"{paper.metric_value}" if paper.metric_value != UNKNOWN else UNKNOWN
    )

    caveat_lines = "\n".join(f"  - ⚠️ {c}" for c in caveats) if caveats else "  - None"

    _cat_icon = {
        "STRONG SOTA CANDIDATE": "🟢",
        "WEAK SOTA CANDIDATE": "🟡",
        "RELATED WORK ONLY": "🔴",
    }
    cat_icon = _cat_icon.get(paper.sota_category, "⚪")
    cat_line = f"{cat_icon} **{paper.sota_category}**"
    if isinstance(paper.confidence_score, float):
        cat_line += f"  ·  Confidence score: {paper.confidence_score:.2f}"

    lines = [
        f"### {idx}. {title_md}",
        "",
        cat_line,
        "",
        f"- **Year:** {paper.year}",
        f"- **Dataset(s):** {datasets}",
        f"- **Task:** {paper.task}",
        f"- **Label granularity:** {paper.label_granularity}",
        f"- **Modalities:** {modalities}",
        f"- **Preprocessing:** {preprocessing}",
        f"- **Split:** {split_str}",
        f"- **CV:** {paper.cross_validation}",
        f"- **Split type:** {paper.split_type}",
        f"- **Model:** {paper.model_name}",
        f"- **Baselines compared:** {baselines}",
        f"- **Metric:** {paper.metric_name} = **{metric_str}**",
        f"- **Result scope:** {paper.result_scope}",
        f"- **Reviewer confidence:** {paper.confidence}",
    ]

    if paper.notes:
        lines.append(f"- **Notes:** {paper.notes}")

    if paper.source_snippet:
        lines.append(f'- **Source:** *"{paper.source_snippet}"*')

    lines += ["- **Caveats:**", caveat_lines]

    critical_flags = [f for f in paper.red_flags if f.startswith("CRITICAL")]
    secondary_flags = [f for f in paper.red_flags if f.startswith("SECONDARY")]
    if critical_flags or secondary_flags:
        flag_lines = []
        for flag in critical_flags:
            parts = flag.split("|", 2)
            flag_lines.append(f"  - 🚨 **[Critical — {parts[1]}]** {parts[2]}")
        for flag in secondary_flags:
            parts = flag.split("|", 2)
            flag_lines.append(f"  - ⚠️ **[{parts[1]}]** {parts[2]}")
        lines += ["- **Red flags:**"] + flag_lines
    else:
        lines.append("- **Red flags:** None detected")

    lines.append("")
    return "\n".join(lines)


def _group_table(group: ComparisonGroup) -> str:
    """Render a markdown table for a comparison group."""
    papers_with_value = [
        p for p in group.papers if p.metric_value != UNKNOWN
    ]
    papers_no_value = [p for p in group.papers if p.metric_value == UNKNOWN]

    rows: List[str] = []
    rows.append(
        f"| # | Title | Year | Model | {group.metric} | Result Scope | Split Type | Confidence |"
    )
    rows.append("|---|-------|------|-------|----------------|--------------|------------|------------|")

    # Sort by metric value descending where available
    def sort_key(p: PaperRecord):
        v = p.metric_value
        return float(v) if isinstance(v, (int, float)) else -1.0

    for i, paper in enumerate(
        sorted(papers_with_value, key=sort_key, reverse=True) + papers_no_value, 1
    ):
        url = paper.paper_url if paper.paper_url != UNKNOWN else None
        title_md = f"[{paper.title}]({url})" if url else paper.title
        val = f"{paper.metric_value}" if paper.metric_value != UNKNOWN else "—"
        rows.append(
            f"| {i} | {title_md} | {paper.year} | {paper.model_name} "
            f"| {val} | {paper.result_scope} | {paper.split_type} | {paper.confidence} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main report writer
# ---------------------------------------------------------------------------

def write_markdown(result: ComparisonResult, papers: List[PaperRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    sections: List[str] = []

    sections.append("# SOTA Finder — Review Report\n")
    sections.append(
        "> This report was generated automatically to **support** manual review.\n"
        "> Do **not** treat any group as a definitive SOTA without verifying the\n"
        "> original papers and their evaluation protocols.\n"
    )

    # Summary counts
    n_comparable = sum(len(g.papers) for g in result.comparable)
    n_partial = sum(len(g.papers) for g in result.partially_comparable)
    n_not = len(result.not_comparable)

    sections.append("## Summary\n")
    sections.append(
        f"| Category | Papers |\n"
        f"|----------|--------|\n"
        f"| Comparable groups | {n_comparable} paper(s) across {len(result.comparable)} group(s) |\n"
        f"| Partially comparable | {n_partial} paper(s) across {len(result.partially_comparable)} group(s) |\n"
        f"| Not comparable | {n_not} paper(s) |\n"
        f"| **Total** | **{len(papers)}** |\n"
    )

    # ---- Comparable groups ----
    sections.append("---\n\n## Comparable Groups\n")
    sections.append(
        "_Papers in these groups share the same task, metric, label granularity, "
        "and split type. They **may** be directly compared, but always verify "
        "evaluation protocols before drawing conclusions._\n"
    )

    if result.comparable:
        for gi, group in enumerate(result.comparable, 1):
            sections.append(
                f"### Group C{gi}: {group.task} | {group.metric} | "
                f"{group.label_granularity} | {group.split_type}\n"
            )
            sections.append(_group_table(group) + "\n")
    else:
        sections.append("_No fully comparable groups found._\n")

    # ---- Partially comparable ----
    sections.append("---\n\n## Partially Comparable\n")
    sections.append(
        "_Papers here share task and metric but differ on label granularity or split type. "
        "**Do not compare scores directly.** Use this section to identify papers worth "
        "re-reading for methodological details._\n"
    )

    if result.partially_comparable:
        for gi, group in enumerate(result.partially_comparable, 1):
            task, metric, granularity, split = group.key
            sections.append(
                f"### Group P{gi}: {task} | {metric} "
                f"(granularity: {granularity}, split: {split})\n"
            )
            sections.append(_group_table(group) + "\n")
    else:
        sections.append("_No partially comparable papers._\n")

    # ---- Not comparable ----
    sections.append("---\n\n## Not Comparable\n")
    sections.append(
        "_Papers here could not be grouped — missing task, metric, or other key fields._\n"
    )

    if result.not_comparable:
        for gi, paper in enumerate(result.not_comparable, 1):
            sections.append(_paper_block(paper, gi))
    else:
        sections.append("_No papers in this category._\n")

    # ---- Full paper details ----
    sections.append("---\n\n## Full Paper Details\n")
    sections.append(
        "_Complete extracted information for every paper, including caveats._\n"
    )

    paper_index: dict[str, int] = {}
    for i, paper in enumerate(papers, 1):
        paper_index[paper.title] = i
        sections.append(_paper_block(paper, i))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections))
