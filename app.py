"""Streamlit web app for sota-finder.

Run with:
    streamlit run app.py

No API keys required — uses only free OpenAlex and Semantic Scholar APIs.
Set GEMINI_API_KEY (free) or ANTHROPIC_API_KEY to enable AI-assisted field extraction.
"""

from __future__ import annotations

import os

import streamlit as st

from src.extract import auto_build_records
from src.fetch_fulltext import enrich_with_fulltext, pdf_text_from_file
from src.fetch_papers import find_dataset_paper
from src.llm_extract import llm_enrich_records
from src.red_flags import apply_red_flags
from src.rule_extract import enrich_records, extract_all_metrics, top_n, score_paper
from src.schemas import UNKNOWN

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SOTA Finder",
    page_icon="🔬",
    layout="wide",
)

st.title("SOTA Finder")
st.caption(
    "Finds candidate SOTA papers for biomedical ML datasets. "
    "Human review required — this tool does not declare a winner."
)

# ---------------------------------------------------------------------------
# Field guide
# ---------------------------------------------------------------------------

with st.expander("📋 Field guide — what to look for when you open the paper"):
    st.markdown("""
| Field | Where to look in the paper |
|---|---|
| **Task** | Title or first sentence of abstract |
| **Metric** | Last sentence of abstract, or Results section header |
| **Value** | Results table — use the *test-set* row; confirm it matches the metric name |
| **Split type** | Methods → Experimental Setup: look for "LOSO", "leave-one-subject-out", or "random split" |
| **Label granularity** | Dataset section: how many classes / stage labels (e.g. 5-class AASM, binary) |
| **Modalities** | Dataset section: which physiological signals were used (EEG, ECG, EOG…) |
| **Cross-validation** | Methods → Evaluation: fold count and strategy (e.g. "10-fold subject-wise", "LOSO") |
| **Model name** | Title or Methods → Architecture section |
| **Preprocessing** | Methods → Signal Processing: filter bands (Hz), sampling rate, epoch/window length |
| **Multi-dataset** | Methods: does the paper pretrain on external data before evaluating on this dataset? |
""")

# ---------------------------------------------------------------------------
# Search bar
# ---------------------------------------------------------------------------

col_acronym, col_fullname, col_btn = st.columns([2, 3, 1])
with col_acronym:
    dataset = st.text_input(
        "Acronym / short name",
        placeholder='e.g. DREAMT, SHHS, FACED, WESAD',
        help="The dataset acronym or short name used in papers (e.g. DREAMT, SHHS, MIT-BIH).",
    )
with col_fullname:
    full_name = st.text_input(
        "Full dataset name (optional but recommended)",
        placeholder='e.g. "Dataset for Real-time EEG-based and AMbulatory Testing"',
        help=(
            "The full expanded name of the dataset. "
            "Providing this greatly improves identification of the original dataset paper, "
            "especially for acronyms that are also common English words (DREAMT, FACED, etc.)."
        ),
    )
with col_btn:
    st.write("")  # vertical align
    search = st.button("Search", use_container_width=True, type="primary")

modalities = st.multiselect(
    "Signal modalities (select all that apply — strongly recommended for ambiguous acronyms)",
    options=[
        "EEG", "ECG / EKG", "fMRI", "ECoG", "EMG",
        "PPG / BVP", "EDA / GSR", "Wearable / IMU / Accelerometer",
        "PSG (polysomnography)", "Eye tracking / EOG", "fNIRS",
        "Audio / speech", "Video / facial", "Multimodal",
    ],
    help=(
        "Used to disambiguate datasets whose acronym matches unrelated papers "
        "(e.g. FACED could be an emotion EEG dataset or a computer vision paper). "
        "Papers that don't mention any selected modality are down-ranked."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOT_FOUND = "*not found*"


def _field_md(label: str, v) -> str:
    """Format a field for markdown display. UNKNOWN renders as italic note."""
    if v == UNKNOWN or v is None:
        return f"**{label}**  \n{_NOT_FOUND}"
    if isinstance(v, float):
        return f"**{label}**  \n{v:.2f}%"
    return f"**{label}**  \n{v}"


def _completeness(p: object) -> tuple[int, int]:
    """Return (filled_count, total_count) for extractable fields."""
    has_split = any(
        getattr(p, f, UNKNOWN) != UNKNOWN
        for f in ("train_split", "val_split", "test_split")
    )
    filled = sum([
        getattr(p, "task", UNKNOWN) != UNKNOWN,
        getattr(p, "metric_name", UNKNOWN) != UNKNOWN,
        isinstance(getattr(p, "metric_value", UNKNOWN), float),
        getattr(p, "model_name", UNKNOWN) != UNKNOWN,
        bool(getattr(p, "modalities", [])),
        getattr(p, "cross_validation", UNKNOWN) != UNKNOWN,
        has_split,
        bool(getattr(p, "preprocessing_steps", [])),
    ])
    return filled, 8


def _notion_text(p) -> str:
    """Format paper fields as plain text for pasting into Notion."""
    mv = f"{p.metric_value:.2f}%" if isinstance(p.metric_value, float) else "not found"
    mods = ", ".join(p.modalities) if p.modalities else "not found"
    prep = ", ".join(p.preprocessing_steps) if p.preprocessing_steps else "not found"
    foundation = {
        "yes": "Foundation model",
        "no": "Not a foundation model",
    }.get(getattr(p, "foundation_model", UNKNOWN), "not found")

    metric_line = p.metric_name if p.metric_name != UNKNOWN else "not found"
    if isinstance(p.metric_value, float):
        metric_line += f" = {mv}"

    split_parts = []
    if p.train_split != UNKNOWN:
        split_parts.append(f"train {p.train_split}")
    if p.val_split != UNKNOWN:
        split_parts.append(f"val {p.val_split}")
    if p.test_split != UNKNOWN:
        split_parts.append(f"test {p.test_split}")
    split_str = " / ".join(split_parts) if split_parts else "not found"

    return "\n".join([
        f"Title: {p.title}",
        f"Year: {p.year if p.year != UNKNOWN else 'unknown'}",
        f"Task: {p.task if p.task != UNKNOWN else 'not found'}",
        f"Metric: {metric_line}",
        f"Model: {p.model_name if p.model_name != UNKNOWN else 'not found'}",
        f"Foundation model tag: {foundation}",
        f"Modalities: {mods}",
        f"Cross-validation: {p.cross_validation if p.cross_validation != UNKNOWN else 'not found'}",
        f"Data split: {split_str}",
        f"Preprocessing: {prep}",
        f"Paper URL: {p.paper_url if p.paper_url != UNKNOWN else 'not found'}",
    ])


def _enrich_paper_with_pdf(p, pdf_text: str, use_llm: bool):
    """Return a paper updated with text extracted from a manually uploaded PDF."""
    enriched = p.model_copy(update={
        "notes": (p.notes or "") + f" FullText: {pdf_text[:25_000]}"
    })
    enriched = enrich_records([enriched])[0]
    enriched = apply_red_flags([enriched])[0]
    if use_llm:
        enriched = llm_enrich_records([enriched])[0]
    return enriched

# ---------------------------------------------------------------------------
# Search & display
# ---------------------------------------------------------------------------

if search and dataset.strip():
    dataset = dataset.strip()
    full_name = full_name.strip() if full_name else ""

    # Step 1: Find and display the original dataset paper
    _spinner_label = (
        f"Step 1 — Finding dataset paper for **{dataset}**"
        + (f" / *{full_name[:60]}*" if full_name else "") + "…"
    )
    with st.spinner(_spinner_label):
        ds_paper = find_dataset_paper(dataset, full_name=full_name or None, modalities=modalities or None)

    if ds_paper.get("found"):
        with st.expander("📄 Dataset paper identified (anchor for all searches)", expanded=True):
            authors_str = ", ".join(ds_paper["authors"][:5])
            if len(ds_paper["authors"]) > 5:
                authors_str += f" +{len(ds_paper['authors']) - 5} more"
            doi_str = ds_paper.get("doi", "UNKNOWN")
            doi_link = (f"[{doi_str}](https://doi.org/{doi_str.lstrip('https://doi.org/')})"
                        if doi_str not in ("UNKNOWN", "") else "UNKNOWN")
            st.markdown(
                f"**{ds_paper['title']}**  \n"
                f"*{authors_str}* · {ds_paper['venue']} · {ds_paper['year']}  \n"
                f"DOI: {doi_link} · **{ds_paper['citation_count']:,} citations**"
            )
            paper_url = ds_paper.get("paper_url", "UNKNOWN")
            if paper_url and paper_url != "UNKNOWN":
                st.markdown(f"[📥 Access / download dataset]({paper_url})")

            # Warn about any identifiers not confirmed in the paper
            missing = ds_paper.get("missing_flags", set())
            if missing:
                missing_labels = {"acronym": "dataset acronym", "full_name": "full dataset name",
                                  "modality": "modality keyword"}
                missing_str = ", ".join(missing_labels.get(f, f) for f in missing)
                st.warning(
                    f"⚠️ Could not confirm **{missing_str}** in the paper's title, abstract, "
                    "or page. It may still be the correct paper — verify manually."
                )

            # Dataset statistics
            stats = ds_paper.get("stats", {})
            stat_items = [
                ("Subjects", stats.get("subjects")),
                ("Channels", stats.get("channels")),
                ("Recording duration", stats.get("duration")),
                ("Dataset size", stats.get("size")),
                ("Files / recordings", stats.get("files")),
                ("Train/test split", stats.get("split")),
            ]
            filled = [(k, v) for k, v in stat_items if v]
            labels_list = stats.get("labels", [])
            if filled or labels_list:
                st.markdown("**Dataset statistics**")
                if filled:
                    cols = st.columns(min(3, len(filled)))
                    for i, (k, v) in enumerate(filled):
                        cols[i % 3].metric(k, v)
                if labels_list:
                    st.markdown(f"**Labels / classes:** {', '.join(labels_list)}")

            pwc_col, _ = st.columns([2, 3])
            if ds_paper["pwc_listed"]:
                pwc_col.success("✓ Listed on PapersWithCode — verified benchmark metrics available")
            else:
                pwc_col.warning(
                    "✗ Not listed on PapersWithCode — SOTA harder to independently verify. "
                    "Treat all reported metrics with extra scrutiny."
                )
    else:
        st.warning(
            f"Could not identify the original dataset paper for **{dataset}**. "
            "Falling back to keyword-only search — results may be less complete."
        )

    with st.spinner(f"Running 5-step targeted search for **{dataset}**…"):
        raw_papers = auto_build_records(dataset, full_name=full_name or None,
                                        modalities=modalities or None, dataset_paper=ds_paper)

    if not raw_papers:
        st.error(
            f"No papers found for **{dataset}**. "
            "Try a different spelling, abbreviation, or check your network connection."
        )
        st.stop()

    with st.spinner("Extracting fields from abstracts (rule-based)…"):
        papers = enrich_records(raw_papers)

    # Fetch full text for top 20 candidates in parallel
    top_candidates = top_n(papers, n=20)
    other_papers = [p for p in papers if p not in top_candidates]

    with st.spinner(
        "Fetching full paper text (arXiv, bioRxiv, Unpaywall…) — "
        "enables Methods-section split-type and preprocessing detection"
    ):
        top_candidates = enrich_with_fulltext(top_candidates)

    # Re-run rule extraction with full text, detect red flags, then rank final top 8
    # Red flags penalise scores so methodologically questionable papers rank lower
    top_candidates = enrich_records(top_candidates)
    top_candidates = apply_red_flags(top_candidates)
    candidates = top_n(top_candidates, n=8)

    _has_llm = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
    if _has_llm:
        _provider = "Gemini" if os.environ.get("GEMINI_API_KEY") else "Claude"
        with st.spinner(f"Enhancing extraction with {_provider} AI…"):
            candidates = llm_enrich_records(candidates)
    else:
        st.caption(
            "💡 Set `GEMINI_API_KEY` in your environment to enable free AI-powered "
            "field extraction (reduces *not found* values significantly). "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    papers = top_candidates + other_papers

    # Summary bar
    n_total = len(papers)
    n_with_metric = sum(1 for p in papers if isinstance(p.metric_value, float))
    n_with_task = sum(1 for p in papers if p.task != UNKNOWN)

    st.markdown(f"### Top {len(candidates)} of {n_total} candidates for **{dataset}**")

    from src.red_flags import STRONG, WEAK, RELATED
    n_strong  = sum(1 for p in candidates if p.sota_category == STRONG)
    n_weak    = sum(1 for p in candidates if p.sota_category == WEAK)
    n_related = sum(1 for p in candidates if p.sota_category == RELATED)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total papers fetched", n_total)
    m2.metric("Papers with numeric metric", n_with_metric)
    m3.metric("🟢 Strong SOTA candidates", n_strong)
    m4.metric("🟡 Weak candidates", n_weak,
              help="Pass some but not all critical checks. Review flags before citing.")
    m5.metric("🔴 Related work only", n_related,
              help="Failed critical methodology checks. Do not rank as SOTA.")

    st.info(
        "Fields are extracted automatically from titles, abstracts, and full paper text where available. "
        "Treat every extracted detail as a lead, not a final answer: open the real PDF or paper "
        "and verify the metric, split, dataset use, preprocessing, and SOTA claim before citing it. "
        "The web app can miss or misread details, including fields it does not mark as *not found*. "
        "Never compare papers across different metrics or split types.",
        icon="⚠️",
    )

    st.divider()

    _CATEGORY_BADGE = {
        "STRONG SOTA CANDIDATE": "🟢 STRONG SOTA CANDIDATE",
        "WEAK SOTA CANDIDATE":   "🟡 WEAK SOTA CANDIDATE",
        "RELATED WORK ONLY":     "🔴 RELATED WORK ONLY",
    }

    if "pdf_enrichments" not in st.session_state:
        st.session_state.pdf_enrichments = {}

    # Paper cards
    for rank, p in enumerate(candidates, start=1):
        # Use PDF-uploaded version if available
        p = st.session_state.get("pdf_enrichments", {}).get(p.title, p)

        filled, total = _completeness(p)
        has_full_text = bool(p.notes and "FullText:" in p.notes)

        with st.container(border=True):
            header_col, bar_col = st.columns([7, 3])

            with header_col:
                if p.paper_url and p.paper_url != UNKNOWN:
                    st.markdown(f"**{rank}. [{p.title}]({p.paper_url})**")
                else:
                    st.markdown(f"**{rank}. {p.title}**")

                if has_full_text:
                    st.caption("📄 Full text read")
                else:
                    st.caption("📋 Abstract only — upload PDF below for deeper extraction")
                    st.info(
                        "Need more information from this paper? Sign in through your "
                        "school, library, or publisher account to access the full text, "
                        "then upload the PDF below.",
                        icon="ℹ️",
                    )

                # Category badge + confidence score
                badge = _CATEGORY_BADGE.get(p.sota_category, "")
                conf_str = (
                    f"  ·  Confidence score: **{p.confidence_score:.2f}**"
                    if isinstance(p.confidence_score, float)
                    else ""
                )
                if badge:
                    st.markdown(f"{badge}{conf_str}")

                year_str = str(p.year) if p.year != UNKNOWN else "Year unknown"
                source = ""
                if p.notes:
                    for src in (
                        "OpenAlex (citation chain)", "PhysioNet (citation chain)",
                        "Semantic Scholar (citation chain)",
                        "PapersWithCode", "PhysioNet",
                        "OpenAlex", "Semantic Scholar",
                    ):
                        if src in p.notes:
                            source = src
                            break
                meta = f"*{year_str}*"
                if source:
                    meta += f"  ·  {source}"
                st.caption(meta)

            with bar_col:
                st.progress(
                    min(filled / total, 1.0),
                    text=f"Completeness: {filled}/{total} fields auto-filled",
                )
                uploaded_pdf = st.file_uploader(
                    "Upload PDF",
                    type=["pdf"],
                    accept_multiple_files=False,
                    key=f"card_pdf_upload_{dataset}_{rank}",
                    help=(
                        "Upload this paper's PDF if you accessed it through a "
                        "school, library, or publisher login."
                    ),
                )
                if uploaded_pdf is not None:
                    file_key = f"{dataset}__{p.title}__{uploaded_pdf.name}"
                    if file_key in st.session_state.pdf_enrichments:
                        st.success("PDF processed")
                    else:
                        with st.spinner("Reading uploaded PDF..."):
                            try:
                                pdf_text = pdf_text_from_file(uploaded_pdf.read())
                            except Exception as e:
                                st.error(f"Could not read PDF: {e}")
                                pdf_text = ""

                        if pdf_text:
                            with st.spinner("Updating this paper..."):
                                enriched = _enrich_paper_with_pdf(p, pdf_text, _has_llm)
                            st.session_state.pdf_enrichments[p.title] = enriched
                            st.session_state.pdf_enrichments[file_key] = enriched
                            st.success("Updated from PDF")
                            st.rerun()
                        else:
                            st.error("Could not extract text from this PDF.")

            # Key metrics row — all numeric metrics found in the text
            paper_text = ""
            if p.notes and "FullText:" in p.notes:
                paper_text = p.notes.split("FullText:", 1)[1].strip()
            elif p.notes and "Abstract:" in p.notes:
                paper_text = p.notes.split("Abstract:", 1)[1].split("FullText:", 1)[0].strip()

            all_metrics = extract_all_metrics(f"{p.title} {paper_text}")
            if all_metrics:
                st.markdown("**Key metrics found**")
                metric_cols = st.columns(min(len(all_metrics), 6))
                for col, (mname, mval) in zip(metric_cols, all_metrics[:6]):
                    col.metric(mname, f"{mval:.1f}%")
                st.divider()

            # Field grid — row 1: task / metric / model / cross-validation
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(_field_md("Task", p.task))

            if p.metric_name != UNKNOWN:
                metric_display = p.metric_name
                if isinstance(p.metric_value, float):
                    metric_display += f" = {p.metric_value:.2f}%"
            else:
                metric_display = UNKNOWN
            c2.markdown(_field_md("Metric (primary)", metric_display))
            c3.markdown(_field_md("Model", p.model_name))
            c4.markdown(_field_md("Cross-validation", p.cross_validation))

            foundation_label = {
                "yes": "Foundation model",
                "no": "Not a foundation model",
            }.get(getattr(p, "foundation_model", UNKNOWN), "Foundation model status unknown")
            st.caption(f"Tag: {foundation_label}")

            # Field grid — row 2: modalities / data split / preprocessing
            c5, c6, c7 = st.columns(3)
            mods_val = ", ".join(p.modalities) if p.modalities else UNKNOWN
            c5.markdown(_field_md("Modalities", mods_val))

            tr, va, te = p.train_split, p.val_split, p.test_split
            has_split = any(x != UNKNOWN for x in (tr, va, te))
            if has_split:
                parts = []
                if tr != UNKNOWN:
                    parts.append(f"train {tr}")
                if va != UNKNOWN:
                    parts.append(f"val {va}")
                if te != UNKNOWN:
                    parts.append(f"test {te}")
                split_str = " / ".join(parts)
            else:
                split_str = UNKNOWN
            c6.markdown(_field_md("Data split", split_str))

            prep_val = ", ".join(p.preprocessing_steps) if p.preprocessing_steps else UNKNOWN
            c7.markdown(_field_md("Preprocessing", prep_val))

            # Caveats — inline warnings
            caveats: list[str] = []
            if p.target_dataset_only is False:
                caveats.append("Uses multiple datasets or pretraining — scores may not be comparable to single-dataset papers")
            if p.result_scope not in ("test", UNKNOWN):
                caveats.append(f"Result scope: {p.result_scope} (not test set — check if a test-set number is reported)")
            if p.confidence == "low":
                caveats.append("Low confidence — methodology unclear from paper")

            for cav in caveats:
                st.warning(cav, icon="⚠️")

            # Red flags — CRITICAL first, then SECONDARY
            critical_flags = [f for f in p.red_flags if f.startswith("CRITICAL")]
            secondary_flags = [f for f in p.red_flags if f.startswith("SECONDARY")]
            for flag in critical_flags:
                parts = flag.split("|", 2)
                st.error(f"🚩 **[Critical — {parts[1]}]** {parts[2]}", icon="🚨")
            for flag in secondary_flags:
                parts = flag.split("|", 2)
                st.warning(f"⚑ **[{parts[1]}]** {parts[2]}", icon="⚠️")

            # Copy for Notion
            with st.expander("📋 Copy for Notion"):
                st.code(_notion_text(p), language=None)

            # Full abstract
            if p.notes and "Abstract:" in p.notes:
                abstract_text = p.notes.split("Abstract:", 1)[1].split("FullText:", 1)[0].strip()
                if abstract_text:
                    with st.expander("Abstract"):
                        st.write(abstract_text)

    # ---------------------------------------------------------------------------
    # PDF upload — for papers behind your institution's paywall
    # ---------------------------------------------------------------------------
    st.divider()
    with st.expander("📄 Upload paywalled papers (school / library access)", expanded=False):
        st.info(
            "For more complete extraction, sign in through your school, library, "
            "or publisher account before downloading paywalled papers.",
            icon="ℹ️",
        )
        st.markdown(
            "If a paper above is behind a paywall that your institution can access, "
            "download the PDF and upload it here. The tool will re-extract all fields "
            "from the full text and update that paper's card above."
        )

        uploaded_pdfs = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_upload_{dataset}",
        )

        for uf in (uploaded_pdfs or []):
            file_key = f"{dataset}__{uf.name}"
            if file_key in st.session_state.pdf_enrichments:
                st.success(f"✅ **{uf.name}** already processed — see updated card above.")
                continue

            with st.spinner(f"Extracting text from **{uf.name}**…"):
                try:
                    pdf_bytes = uf.read()
                    pdf_text = pdf_text_from_file(pdf_bytes)
                except Exception as e:
                    st.error(f"Could not read {uf.name}: {e}")
                    continue

            if not pdf_text:
                st.error(
                    f"Could not extract text from **{uf.name}**. "
                    "Make sure it is a text-based PDF (not a scanned image)."
                )
                continue

            # Auto-match: extract first non-empty line as title guess
            title_guess = next(
                (ln.strip() for ln in pdf_text.splitlines() if len(ln.strip()) > 10),
                ""
            )

            import difflib
            candidate_titles = [p.title for p in candidates]
            close = difflib.get_close_matches(
                title_guess.lower(),
                [t.lower() for t in candidate_titles],
                n=1, cutoff=0.35,
            )
            auto_idx = (
                [t.lower() for t in candidate_titles].index(close[0]) + 1
                if close else 0
            )

            selected = st.selectbox(
                f"Match **{uf.name}** to which paper?",
                options=["— select paper —"] + candidate_titles,
                index=auto_idx,
                key=f"match__{file_key}",
            )

            if selected == "— select paper —":
                st.info("Select the paper this PDF belongs to, then click Process.")
                continue

            if st.button(f"Process '{uf.name}'", key=f"btn__{file_key}"):
                matched = next(p for p in candidates if p.title == selected)
                with st.spinner("Re-extracting fields from full text…"):
                    enriched = _enrich_paper_with_pdf(matched, pdf_text, _has_llm)

                st.session_state.pdf_enrichments[selected] = enriched
                # Also cache under the file key so we don't reprocess
                st.session_state.pdf_enrichments[file_key] = enriched
                st.success(
                    f"✅ Updated **{selected}** with full text from **{uf.name}**. "
                    "Scroll up to see the refreshed card."
                )
                st.rerun()

    st.divider()
    st.caption(
        "Sources: [OpenAlex](https://openalex.org) · "
        "[Semantic Scholar](https://www.semanticscholar.org)  ·  "
        "All fields extracted from abstracts only — verify before citing."
    )

elif search and not dataset.strip():
    st.warning("Please enter a dataset name first.")
