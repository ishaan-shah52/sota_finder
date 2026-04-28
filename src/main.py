"""Entry point: python -m src.main [--dataset <name>]"""

from __future__ import annotations

import argparse
import os

from src.compare import group_papers
from src.red_flags import apply_red_flags
from src.render_report import write_csv, write_markdown

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def _print_summary(result, papers, md_path: str = "") -> None:
    n_comparable = sum(len(g.papers) for g in result.comparable)
    n_partial = sum(len(g.papers) for g in result.partially_comparable)
    n_not = len(result.not_comparable)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"  Comparable groups  : {len(result.comparable)} group(s), "
        f"{n_comparable} paper(s)"
    )
    for g in result.comparable:
        print(f"    [{g.task} | {g.metric} | {g.split_type}]")
        for p in g.papers:
            print(f"      - {p.title[:55]}")
    print()
    print(
        f"  Partially comparable: {len(result.partially_comparable)} group(s), "
        f"{n_partial} paper(s)"
    )
    for g in result.partially_comparable:
        task, metric, granularity, split = g.key
        print(f"    [{task} | {metric} | granularity={granularity} | split={split}]")
        for p in g.papers:
            print(f"      - {p.title[:55]}")
    print()
    print(f"  Not comparable     : {n_not} paper(s)")
    for p in result.not_comparable:
        print(f"      - {p.title[:55]}")
    print("=" * 60)
    if md_path:
        print(f"\nReview {md_path} to manually assess SOTA candidates.")
    else:
        print("\nReview outputs/report.md to manually assess SOTA candidates.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sota-finder: find and review SOTA papers for a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="NAME",
        help='Dataset name to search (e.g. "SHHS", "MIT-BIH"). '
             "If omitted, runs on built-in mock data.",
    )
    args = parser.parse_args()

    if args.dataset:
        from src.extract import auto_build_records
        papers = auto_build_records(args.dataset)
        if not papers:
            print(
                f"\nNo papers found for dataset {args.dataset!r}. "
                "Try a different name or check network connectivity."
            )
            return
        print(f"\nLoaded {len(papers)} paper record(s) for {args.dataset!r}.")
        # Per-dataset output directory so runs don't overwrite each other
        safe_name = args.dataset.replace(" ", "_").replace("/", "-")
        out_dir = os.path.join(OUTPUT_DIR, safe_name)
    else:
        from src.mock_data import MOCK_PAPERS
        papers = MOCK_PAPERS
        print(f"No --dataset provided. Running on mock data ({len(papers)} records).")
        out_dir = OUTPUT_DIR

    csv_path = os.path.join(out_dir, "report.csv")
    md_path = os.path.join(out_dir, "report.md")

    papers = apply_red_flags(papers)
    result = group_papers(papers)

    write_csv(papers, csv_path)
    print(f"CSV written -> {csv_path}")

    write_markdown(result, papers, md_path)
    print(f"Markdown report written -> {md_path}")

    _print_summary(result, papers, md_path)


if __name__ == "__main__":
    main()
