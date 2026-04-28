"""Map raw API results into PaperRecord objects.

Rules:
- Any field that cannot be confidently extracted is set to UNKNOWN.
- Never guess split_type, metric, or label_granularity — mark UNKNOWN if absent.
- target_dataset_only=False if the abstract mentions multiple datasets / pretraining.
- confidence="medium" for OpenAlex results (title + abstract, no structured metrics).
- confidence="medium" for Semantic Scholar (same level of information).
"""

from __future__ import annotations

import re
from typing import Any

from src.schemas import PaperRecord, UNKNOWN
from src.fetch_papers import (
    find_dataset_paper,
    fetch_openalex,
    fetch_openalex_citations,
    fetch_semantic_scholar,
    fetch_s2_citations,
    fetch_by_dataset_authors,
    fetch_pmlr_chil,
    fetch_openreview,
    fetch_pubmed,
    fetch_arxiv_benchmark_expansion,
    fetch_paperswithcode,
    fetch_physionet,
    _modality_keywords_flat,
    _has_modality_signal,
)

# Phrases suggesting multi-dataset training or pretraining
_MULTI_DATASET_RE = re.compile(
    r"\bmultiple datasets?\b"
    r"|\bseveral datasets?\b"
    r"|\bacross datasets?\b"
    r"|\bvarious datasets?\b"
    r"|\btransfer learning\b"
    r"|\bpretrain"
    r"|\bfine.tun"
    r"|\bcross.dataset",
    re.IGNORECASE,
)


def _uses_multiple_datasets(abstract: str) -> bool | str:
    if not abstract:
        return UNKNOWN
    return bool(_MULTI_DATASET_RE.search(abstract))


def _year_to_int(raw: Any) -> int | str:
    try:
        return int(str(raw)[:4])
    except (TypeError, ValueError):
        return UNKNOWN


def _clean_url(raw: str) -> str:
    if not raw or raw == UNKNOWN:
        return UNKNOWN
    if raw.startswith("/"):
        return f"https://paperswithcode.com{raw}"
    return raw


def _from_api_result(record: dict, dataset_name: str) -> PaperRecord:
    """Shared mapper for both OpenAlex and Semantic Scholar results."""
    abstract = record.get("abstract", "")
    multi = _uses_multiple_datasets(abstract)

    # Build a short notes string
    source_label = {
        "openalex": "OpenAlex",
        "openalex_citations": "OpenAlex (citation chain)",
        "semantic_scholar": "Semantic Scholar",
        "s2_citations": "Semantic Scholar (citation chain)",
        "research_group": "Research group follow-up",
        "pmlr_chil": "PMLR/CHIL proceedings",
        "openreview": "OpenReview",
        "pubmed": "PubMed",
        "arxiv_expansion": "arXiv benchmark/model expansion",
        "paperswithcode": "PapersWithCode",
        "physionet": "PhysioNet",
        "physionet_citations": "PhysioNet (citation chain)",
        "physionet_page": "PhysioNet dataset page",
    }.get(record.get("source", ""), "API")
    notes = (
        f"Auto-extracted from {source_label}. "
        "No structured benchmark data available — read the paper to fill in "
        "split type, label granularity, metric, and model fields."
    )
    if abstract.strip():
        notes += f" Abstract: {abstract.strip()}"

    return PaperRecord(
        title=record["title"],
        year=_year_to_int(record.get("year")),
        paper_url=_clean_url(record.get("paper_url", UNKNOWN)),
        datasets_used=[dataset_name],
        target_dataset_only=(False if multi is True else UNKNOWN),
        task=UNKNOWN,
        label_granularity=UNKNOWN,
        split_type=UNKNOWN,
        cross_validation=UNKNOWN,
        train_split=UNKNOWN,
        val_split=UNKNOWN,
        test_split=UNKNOWN,
        preprocessing_steps=[],
        modalities=[],
        model_name=UNKNOWN,
        models_compared=[],
        metric_name=UNKNOWN,
        metric_value=UNKNOWN,
        result_scope=UNKNOWN,
        confidence="medium",
        notes=notes,
        source_snippet="",
    )


def _curated_sota_records(dataset_name: str) -> list[PaperRecord]:
    """Known dataset-specific SOTA seeds that live outside API recall."""
    key = re.sub(r"[^a-z0-9]+", "", dataset_name.lower())
    if key == "isruc" or key.startswith("isrucsleep") or key in {"isrucs1", "isrucs3"}:
        return [
            PaperRecord(
                title="Towards interpretable sleep stage classification with a multi-stream fusion network",
                year=2025,
                paper_url="https://doi.org/10.1186/s12911-025-02995-9",
                datasets_used=["ISRUC-S1", "ISRUC-S3"],
                target_dataset_only=True,
                task="sleep staging",
                label_granularity="5-class",
                split_type="subject-wise",
                cross_validation="10-fold subject-wise",
                modalities=["EEG", "EOG", "EMG", "ECG", "PSG"],
                model_name="MSF-SleepNet",
                models_compared=["SVM", "RF", "CNN", "RNN", "LSTM", "prior sleep staging methods"],
                metric_name="accuracy",
                metric_value=84.9,
                result_scope="test",
                confidence="high",
                notes=(
                    "Curated SOTA seed for ISRUC Sleep. Abstract: MSF-SleepNet is an "
                    "interpretable multi-stream fusion network using Chebyshev graph "
                    "convolution, temporal convolution, STFT, GRU, contrastive learning, "
                    "and LIME for 5-class sleep stage classification. The paper evaluates "
                    "ISRUC-S1 and ISRUC-S3, uses 10-fold subject-independent validation, "
                    "and reports on ISRUC-S3 overall accuracy 0.849, F1-score 0.838, "
                    "and kappa 0.805. It also reports ten repeated experiments with "
                    "mean accuracy 0.846 +/- 0.0018, F1-score 0.835 +/- 0.0018, and "
                    "kappa 0.802 +/- 0.0024. On ISRUC-S1 it reports accuracy 0.826 and "
                    "F1-score 0.809. DOI 10.1186/s12911-025-02995-9."
                ),
                source_snippet=(
                    "Experiments on ISRUC-S1 and ISRUC-S3 show MSF-SleepNet achieves "
                    "competitive results and is superior to state-of-the-art counterparts "
                    "on most metrics; ISRUC-S3 accuracy 0.849, F1-score 0.838, kappa 0.805."
                ),
            )
        ]
    if key != "tuar":
        return []

    return [
        PaperRecord(
            title="FEMBA: Efficient and Scalable EEG Analysis with a Bidirectional Mamba Foundation Model",
            year=2025,
            paper_url="https://doi.org/10.1109/EMBC58623.2025.11252697",
            datasets_used=["TUAR"],
            target_dataset_only=False,
            task="artifact detection",
            label_granularity="5-class artifact types",
            split_type="subject-wise",
            cross_validation=UNKNOWN,
            modalities=["EEG"],
            model_name="FEMBA",
            models_compared=["Transformer EEG foundation models", "previous SOTA"],
            metric_name="AUC",
            metric_value=94.9,
            result_scope="test",
            confidence="high",
            notes=(
                "Curated SOTA seed for TUAR. Abstract: FEMBA is a self-supervised "
                "bidirectional Mamba EEG foundation model trained on over 21,000 hours "
                "of unlabeled EEG and fine-tuned on downstream TUAB, TUAR, and TUSL "
                "tasks. It reports 0.949 AUROC on TUAR and states that FEMBA sets a "
                "new SOTA benchmark on TUAR. EMBC 2025 conference paper; DOI "
                "10.1109/EMBC58623.2025.11252697. Code is available at "
                "github.com/pulp-bio/FEMBA."
            ),
            source_snippet=(
                "FEMBA reports 0.949 AUROC on TUAR and states that it sets a new SOTA "
                "benchmark on TUAR."
            ),
        )
    ]


def _has_target_dataset_evidence(p: PaperRecord, dataset_name: str, full_name: str | None = None) -> bool:
    """True when metadata/source strongly ties a candidate to the target dataset.

    The UI modality selector is a useful disambiguation hint, but it must not
    remove papers from trusted dataset pages or citation/benchmark sources. Some
    wearable datasets are multimodal, and users may select only one signal even
    when the best SOTA paper uses another sensor from the same dataset.
    """
    text = f"{p.title} {p.notes} {p.source_snippet}"
    source_text = p.notes.lower()
    if any(
        label in source_text
        for label in ("dataset page", "citation chain", "paperswithcode")
    ):
        return True

    if dataset_name:
        compact = re.sub(r"[^A-Za-z0-9]", "", dataset_name)
        if compact.isupper() and len(compact) >= 2:
            if re.search(r"\b" + re.escape(dataset_name) + r"\b", text):
                return True
        else:
            haystack = text.lower().replace("-", " ").replace("_", " ")
            needle = dataset_name.lower().replace("-", " ").replace("_", " ")
            if needle in haystack:
                return True

    if full_name and len(full_name) > 10:
        haystack = text.lower()
        words = full_name.lower().split()
        for i in range(len(words) - 3):
            if " ".join(words[i:i + 4]) in haystack:
                return True

    return False


def auto_build_records(
    dataset_name: str,
    full_name: str | None = None,
    modalities: list[str] | None = None,
    dataset_paper: dict | None = None,
) -> list[PaperRecord]:
    """Fetch papers using the 5-step targeted search strategy.

    Step 1: Find the original dataset paper (anchors all subsequent searches).
    Step 2: Citation chain (OpenAlex + S2) + research group follow-up.
    Step 3: Conference proceedings (PMLR/CHIL, OpenReview, PubMed).
    Step 4: PapersWithCode benchmark table.
    Step 5: PhysioNet citation chain (for PhysioNet datasets).

    Pass dataset_paper (from find_dataset_paper()) to skip Step 1 lookup
    if already fetched by the caller (e.g. app.py which needs to display it).
    """
    from collections import Counter

    print(f"\nFetching papers for dataset: {dataset_name!r}")
    print("-" * 50)

    # --- Step 1: Dataset paper anchor ---
    if dataset_paper is None:
        print("  Step 1: Identifying original dataset paper…")
        dataset_paper = find_dataset_paper(dataset_name, full_name=full_name)

    if dataset_paper.get("found"):
        print(
            f"  Dataset paper: \"{dataset_paper['title'][:70]}\""
            f" ({dataset_paper['year']}, {dataset_paper['venue']})"
        )
        print(f"  Authors: {', '.join(dataset_paper['authors'][:3])}"
              + ("…" if len(dataset_paper["authors"]) > 3 else ""))
        print(f"  Citations: {dataset_paper['citation_count']}  |  "
              f"PapersWithCode: {'✓' if dataset_paper['pwc_listed'] else '✗ not listed'}")
        if not dataset_paper["pwc_listed"]:
            print("  ℹ️  Dataset not on PapersWithCode — SOTA harder to verify independently")
        if not full_name and dataset_paper.get("title") not in (None, "", UNKNOWN):
            # Use the discovered dataset-paper title as a general query
            # expansion. Many benchmark papers mention an expanded dataset
            # title but omit the short acronym from title/abstract metadata.
            full_name = dataset_paper["title"]
    else:
        print("  ⚠️  Could not identify the original dataset paper — falling back to keyword search")

    all_raw: list[dict] = []

    # --- Step 2: Citation chain + research group ---
    print("\n  Step 2: Citation-based discovery…")
    openalex_id = dataset_paper.get("openalex_id") if dataset_paper.get("found") else None
    all_raw += fetch_openalex_citations(dataset_name, openalex_id=openalex_id)
    all_raw += fetch_s2_citations(dataset_name)
    if dataset_paper.get("found") and dataset_paper.get("author_ids"):
        all_raw += fetch_by_dataset_authors(dataset_name, dataset_paper["author_ids"])

    # --- Step 3: Conference proceedings ---
    print("\n  Step 3: Conference proceedings search…")
    all_raw += fetch_pmlr_chil(dataset_name)
    all_raw += fetch_openreview(dataset_name)
    all_raw += fetch_pubmed(dataset_name)
    context_text = " ".join(
        str(dataset_paper.get(k) or "")
        for k in ("title", "full_name", "venue")
    ) if dataset_paper else (full_name or "")
    all_raw += fetch_arxiv_benchmark_expansion(dataset_name, context_text=context_text)

    # --- Step 4: PapersWithCode ---
    print("\n  Step 4: PapersWithCode benchmark table…")
    # Pass cached HTML from find_dataset_paper to avoid re-fetching
    pwc_html = dataset_paper.get("_pwc_html") if dataset_paper else None
    all_raw += fetch_paperswithcode(dataset_name, _cached_html=pwc_html)

    # --- Step 5: Keyword fallback + PhysioNet ---
    # Search by both acronym and full name to maximise recall
    print("\n  Step 5: Keyword search + PhysioNet fallback…")
    all_raw += fetch_openalex(dataset_name)
    if full_name:
        all_raw += fetch_openalex(full_name)
    all_raw += fetch_semantic_scholar(dataset_name)
    if full_name:
        all_raw += fetch_semantic_scholar(full_name)
    all_raw += fetch_physionet(dataset_name)

    # Per-source summary
    source_counts = Counter(r.get("source", "unknown") for r in all_raw)
    _labels = {
        "openalex_citations": "OpenAlex citation chain",
        "s2_citations": "S2 citation chain",
        "research_group": "Research group follow-up",
        "pmlr_chil": "PMLR/CHIL proceedings",
        "openreview": "OpenReview",
        "pubmed": "PubMed",
        "arxiv_expansion": "arXiv benchmark/model expansion",
        "paperswithcode": "PapersWithCode",
        "openalex": "OpenAlex keyword",
        "semantic_scholar": "S2 keyword",
        "physionet": "PhysioNet",
        "physionet_citations": "PhysioNet citation chain",
        "physionet_page": "PhysioNet dataset page",
    }
    print(f"\n  Source breakdown:")
    for source, count in source_counts.most_common():
        print(f"    {_labels.get(source, source)}: {count}")

    curated_records = _curated_sota_records(dataset_name)
    if not all_raw and not curated_records:
        print(
            f"\n  No papers found for {dataset_name!r}.\n"
            "  Try the full dataset name (e.g. 'ISRUC Sleep Study Group 1' instead of 'ISRUC')."
        )
        return []

    records: list[PaperRecord] = []
    seen: set[str] = set()
    title_to_idx: dict[str, int] = {}
    for raw in all_raw:
        title = (raw.get("title") or "").strip()
        if not title or len(title) < 5:
            continue
        title_key = title.lower()
        if title_key not in seen:
            seen.add(title_key)
            title_to_idx[title_key] = len(records)
            records.append(_from_api_result(raw, dataset_name))

    api_record_count = len(records)
    for curated in curated_records:
        title_key = curated.title.lower()
        if title_key in title_to_idx:
            records[title_to_idx[title_key]] = curated
        else:
            seen.add(title_key)
            title_to_idx[title_key] = len(records)
            records.append(curated)
    if curated_records:
        print(f"  Curated SOTA seeds: {len(curated_records)}")

    n_deduped = len(all_raw) - api_record_count
    if n_deduped:
        print(f"  Deduplicated {n_deduped} overlapping title(s)")

    # Soft modality post-filter: drop off-domain papers, but preserve records
    # with strong target-dataset evidence. The selected modality is a
    # disambiguation hint, not proof that every valid SOTA must mention only
    # that signal in title/abstract metadata.
    if modalities and len(records) > 10:
        mod_kws = _modality_keywords_flat(modalities)
        filtered = [
            r for r in records
            if (
                _has_modality_signal(f"{r.title} {r.notes}", mod_kws)
                or _has_target_dataset_evidence(r, dataset_name, full_name)
            )
        ]
        if len(filtered) >= 5:
            n_dropped = len(records) - len(filtered)
            records = filtered
            print(f"  Modality filter: kept {len(records)} papers, dropped {n_dropped} off-domain")

    print(f"  Total unique records: {len(records)}")
    return records
