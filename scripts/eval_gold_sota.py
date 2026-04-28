"""Run live retrieval evals for dataset-paper and SOTA-paper discovery.

This script intentionally treats the gold file as evaluation labels only. It
does not feed expected titles back into query generation or ranking.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.extract import auto_build_records
from src.fetch_papers import find_dataset_paper
from src.red_flags import apply_red_flags
from src.rule_extract import dataset_use_evidence, enrich_records, score_paper, top_n
from src.schemas import PaperRecord, UNKNOWN


DEFAULT_GOLD = ROOT / "tests" / "eval" / "gold_sota_cases.json"


def normalize_title(title: str) -> str:
    title = title.lower()
    title = title.replace("&", " and ")
    title = re.sub(r"(?<=\d),(?=\d)", "", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def title_matches(predicted: str, expected: str) -> bool:
    pred = normalize_title(predicted)
    exp = normalize_title(expected)
    if not pred or not exp:
        return False
    if pred == exp or pred in exp or exp in pred:
        return True
    pred_tokens = set(pred.split())
    exp_tokens = set(exp.split())
    if not pred_tokens or not exp_tokens:
        return False
    overlap = len(pred_tokens & exp_tokens)
    return overlap / min(len(pred_tokens), len(exp_tokens)) >= 0.65


def rank_of(candidates: list[tuple[str, float, str]], expected: str) -> int | None:
    for idx, (title, _score, _source) in enumerate(candidates, start=1):
        if title_matches(title, expected):
            return idx
    return None


def format_candidate(idx: int, title: str, score: float, source: str, evidence: str = "") -> str:
    evidence_part = f" evidence={evidence}" if evidence else ""
    return f"  {idx:>2}. score={score:>6.2f} source={source:<32}{evidence_part} {title}"


def failure_reason(rank: int | None, candidates: list[tuple[str, float, str]]) -> str:
    if rank is None:
        if not candidates:
            return "not retrieved"
        return "retrieved candidates did not include expected title"
    if rank > 5:
        return f"expected title retrieved but ranked too low ({rank})"
    return "pass"


def dataset_candidates_from_selected(ds_paper: dict) -> list[tuple[str, float, str]]:
    if ds_paper.get("_candidate_ranking"):
        return [
            (
                str(c.get("title") or "UNKNOWN"),
                float(c.get("score") or 0.0),
                str(c.get("source") or "UNKNOWN"),
            )
            for c in ds_paper["_candidate_ranking"]
        ]
    if not ds_paper.get("found"):
        return []
    title = ds_paper.get("title") or "UNKNOWN"
    score = float(ds_paper.get("citation_count") or 0)
    return [(title, score, "find_dataset_paper")]


def sota_candidates(dataset: str, ds_paper: dict, top_k: int) -> list[tuple[PaperRecord, float]]:
    raw = auto_build_records(dataset, dataset_paper=ds_paper)
    enriched = enrich_records(raw)
    flagged = apply_red_flags(enriched)
    ranked = top_n(flagged, n=max(top_k, len(flagged)))
    return [(p, score_paper(p)) for p in ranked]


def run_case(case: dict[str, str], top_k: int) -> bool:
    dataset = case["dataset"]
    expected_dataset = case["dataset_paper"]
    expected_sota = case["sota_paper"]

    print("\n" + "=" * 88)
    print(f"DATASET: {dataset}")

    ds_paper = find_dataset_paper(dataset)
    dataset_ranked = dataset_candidates_from_selected(ds_paper)
    ds_rank = rank_of(dataset_ranked, expected_dataset)

    print(f"expected dataset paper: {expected_dataset}")
    print("top predicted dataset-paper candidates:")
    if dataset_ranked:
        for i, (title, score, source) in enumerate(dataset_ranked[:top_k], start=1):
            print(format_candidate(i, title, score, source))
    else:
        print("  <none>")
    print(f"dataset rank: {ds_rank or 'not found'}")
    print(f"dataset failure reason: {failure_reason(ds_rank, dataset_ranked)}")

    try:
        ranked_sota = sota_candidates(dataset, ds_paper, top_k)
    except Exception as exc:
        print(f"expected SOTA paper: {expected_sota}")
        print("top predicted SOTA candidates:")
        print(f"  <pipeline error: {exc}>")
        print("SOTA rank: not found")
        print(f"SOTA failure reason: pipeline error: {type(exc).__name__}")
        return False

    sota_printable = [
        (
            p.title,
            score,
            (p.notes.split(". ", 1)[0] if p.notes else UNKNOWN),
            dataset_use_evidence(p),
        )
        for p, score in ranked_sota
    ]
    sota_rank = rank_of([(t, s, src) for t, s, src, _ev in sota_printable], expected_sota)

    print(f"expected SOTA paper: {expected_sota}")
    print("top predicted SOTA candidates:")
    if sota_printable:
        for i, (title, score, source, evidence) in enumerate(sota_printable[:top_k], start=1):
            print(format_candidate(i, title, score, source, evidence))
    else:
        print("  <none>")
    print(f"SOTA rank: {sota_rank or 'not found'}")
    print(f"SOTA failure reason: {failure_reason(sota_rank, sota_printable)}")

    return bool(ds_rank and ds_rank <= top_k and sota_rank and sota_rank <= top_k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate live SOTA Finder retrieval.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", help="Run only one dataset name from the gold file.")
    args = parser.parse_args()

    cases = json.loads(args.gold.read_text(encoding="utf-8"))
    if args.dataset:
        cases = [case for case in cases if case["dataset"].lower() == args.dataset.lower()]
        if not cases:
            print(f"No gold case found for dataset {args.dataset!r}")
            return 2
    results = [run_case(case, args.top_k) for case in cases]
    passed = sum(results)
    print("\n" + "=" * 88)
    print(f"SUMMARY: {passed}/{len(results)} cases passed @ top-{args.top_k}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
