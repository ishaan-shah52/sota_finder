"""LLM-assisted field extraction.

Provider priority:
  1. GEMINI_API_KEY set  → Google Gemini 2.0 Flash (free tier, default)
  2. ANTHROPIC_API_KEY set → Claude Haiku (paid, higher quality)
  3. Neither set          → skip silently, return rule-based results only

Gemini free tier: 15 RPM, 1 M tokens/day — sufficient for any normal search.
"""

from __future__ import annotations

import json
import os

from src.schemas import PaperRecord, UNKNOWN

_SYSTEM = """\
You extract structured metadata from biomedical ML papers. Each paper may include \
title, abstract, and full paper text. Read all provided content carefully.

For EACH paper given, return exactly one JSON object in a JSON array. Fields:
{
  "task": "e.g. sleep staging / seizure detection / arrhythmia classification / \
emotion recognition / motor imagery classification — or UNKNOWN",
  "label_granularity": "e.g. 5-class / binary / 4-class — or UNKNOWN",
  "split_type": "subject-wise | random-window | mixed | UNKNOWN",
  "metric_name": "e.g. accuracy / F1 / macro-F1 / kappa / AUC — or UNKNOWN",
  "metric_value": <number 0-100 representing percentage, or "UNKNOWN">,
  "model_name": "e.g. Transformer / EEGNet / LSTM / CNN / ResNet / SVM — or UNKNOWN",
  "modalities": ["EEG","ECG","EMG","PPG","fMRI","ECoG","EDA","IMU","PSG"] \
(subset that applies, or []),
  "cross_validation": "e.g. 10-fold subject-wise / LOSO / leave-one-subject-out \
— or UNKNOWN",
  "preprocessing_steps": ["bandpass 0.5-40 Hz","ICA","2-second window","STFT",\
"CSP",...] (list of steps, or []),
  "train_split": "e.g. 80% or UNKNOWN",
  "val_split": "e.g. 10% or UNKNOWN",
  "test_split": "e.g. 10% or UNKNOWN",
  "source_snippet": "exact sentence from the text that supports the metric_value, \
or UNKNOWN"
}

Rules:
- split_type "subject-wise": leave-one-subject-out, LOSO, cross-subject, \
patient-independent, inter-patient
- split_type "random-window": random split of windows/epochs/segments (not \
subjects) — scores likely inflated
- metric_value: convert to percentage (0.873 → 87.3, 87.3% → 87.3). \
Report the primary/best test-set result only.
- source_snippet: quote the exact sentence containing metric_value. \
If you cannot find one, return UNKNOWN. Never fabricate a quote.
- Output ONLY the JSON array. No preamble, no explanation, no markdown fences.
- Return exactly as many objects as there are papers, in the same order.
"""


def _build_prompt(to_enrich: list[tuple[int, PaperRecord, str]]) -> str:
    parts = ["Extract metadata from each paper below.\n\n"]
    for rank, (_, p, text) in enumerate(to_enrich, 1):
        limit = 6000 if (p.notes and "FullText:" in p.notes) else 3000
        parts.append(f"Paper {rank}:\nTitle: {p.title}\n{text[:limit]}")
        if rank < len(to_enrich):
            parts.append("\n\n---\n\n")
    return "".join(parts)


def _parse_response(raw: str) -> list[dict] | None:
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


def _call_gemini(prompt: str, api_key: str) -> list[dict] | None:
    try:
        import google.generativeai as genai  # noqa: PLC0415
    except ImportError:
        print("  [LLM Extract] google-generativeai not installed — run: pip install google-generativeai")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=_SYSTEM,
    )
    response = model.generate_content(prompt)
    return _parse_response(response.text)


def _call_claude(prompt: str, api_key: str) -> list[dict] | None:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        print("  [LLM Extract] anthropic not installed — run: pip install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8192,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    return _parse_response(raw)


def llm_enrich_records(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Return a new list with LLM-extracted fields merged in (UNKNOWN fields only).

    Tries Gemini (free) first, then Claude, then returns originals unchanged.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not gemini_key and not anthropic_key:
        return papers

    to_enrich: list[tuple[int, PaperRecord, str]] = []
    for i, p in enumerate(papers):
        body = ""
        if p.notes and "FullText:" in p.notes:
            body = p.notes.split("FullText:", 1)[1].strip()
        elif p.notes and "Abstract:" in p.notes:
            body = p.notes.split("Abstract:", 1)[1].strip()
        full_text = f"{p.title}. {body}".strip()
        if full_text:
            to_enrich.append((i, p, full_text))

    if not to_enrich:
        return papers

    prompt = _build_prompt(to_enrich)

    extracted: list[dict] | None = None
    provider = ""

    if gemini_key:
        try:
            extracted = _call_gemini(prompt, gemini_key)
            provider = "Gemini 2.0 Flash"
        except Exception as exc:
            print(f"  [LLM Extract] Gemini failed ({exc}), trying Claude…")

    if extracted is None and anthropic_key:
        try:
            extracted = _call_claude(prompt, anthropic_key)
            provider = "Claude Haiku"
        except Exception as exc:
            print(f"  [LLM Extract] Claude failed ({exc}) — skipping LLM enrichment")

    if extracted is None:
        print("  [LLM Extract] No response from any provider — skipping")
        return papers

    if _parse_response is None or not isinstance(extracted, list):
        print("  [LLM Extract] Could not parse JSON array — skipping")
        return papers

    result = list(papers)
    for rank, (paper_idx, paper, _) in enumerate(to_enrich):
        if rank >= len(extracted):
            break
        ext = extracted[rank]
        updates: dict = {}

        def _val(v: object) -> bool:
            return v not in (None, "", UNKNOWN)

        if paper.task == UNKNOWN and _val(ext.get("task")):
            updates["task"] = str(ext["task"])

        if paper.label_granularity == UNKNOWN and _val(ext.get("label_granularity")):
            updates["label_granularity"] = str(ext["label_granularity"])

        if paper.split_type == UNKNOWN and ext.get("split_type") in (
            "subject-wise", "random-window", "mixed"
        ):
            updates["split_type"] = ext["split_type"]

        if paper.metric_name == UNKNOWN and _val(ext.get("metric_name")):
            updates["metric_name"] = str(ext["metric_name"])

        if paper.metric_value == UNKNOWN:
            mv = ext.get("metric_value")
            if isinstance(mv, (int, float)) and 0 < mv <= 100:
                updates["metric_value"] = float(mv)

        if paper.model_name == UNKNOWN and _val(ext.get("model_name")):
            updates["model_name"] = str(ext["model_name"])

        if not paper.modalities and isinstance(ext.get("modalities"), list):
            mods = [m for m in ext["modalities"] if isinstance(m, str) and m]
            if mods:
                updates["modalities"] = mods

        if paper.cross_validation == UNKNOWN and _val(ext.get("cross_validation")):
            updates["cross_validation"] = str(ext["cross_validation"])

        if not paper.preprocessing_steps and isinstance(ext.get("preprocessing_steps"), list):
            steps = [s for s in ext["preprocessing_steps"] if isinstance(s, str) and s]
            if steps:
                updates["preprocessing_steps"] = steps

        if paper.train_split == UNKNOWN and _val(ext.get("train_split")):
            updates["train_split"] = str(ext["train_split"])
        if paper.val_split == UNKNOWN and _val(ext.get("val_split")):
            updates["val_split"] = str(ext["val_split"])
        if paper.test_split == UNKNOWN and _val(ext.get("test_split")):
            updates["test_split"] = str(ext["test_split"])

        # Store the grounding quote so the UI can display it
        snippet = ext.get("source_snippet", UNKNOWN)
        if _val(snippet) and not paper.source_snippet:
            updates["source_snippet"] = str(snippet)

        if updates:
            result[paper_idx] = paper.model_copy(update=updates)

    n_enriched = sum(1 for i, p in enumerate(result) if p is not papers[i])
    print(f"  [LLM Extract] Enriched {n_enriched}/{len(to_enrich)} paper(s) via {provider}")
    return result
