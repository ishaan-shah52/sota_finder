"""Fetch candidate SOTA papers from multiple sources.

5-step targeted search strategy:
  Step 1 — find_dataset_paper(): identify the original dataset paper (authors, venue, DOI)
  Step 2 — fetch_openalex_citations(), fetch_s2_citations(): citation-based discovery
           fetch_by_dataset_authors(): follow-up work from the same research group
  Step 3 — fetch_pmlr_chil(), fetch_openreview(), fetch_pubmed(): conference proceedings
  Step 4 — fetch_paperswithcode(): curated benchmark table
  Step 5 — fetch_physionet(): for PhysioNet datasets, DOI citation chain

Citation-based sources catch papers that mention the dataset only in Methods —
missed entirely by abstract-level keyword matching.

Note: PapersWithCode REST API (paperswithcode.com/api/v1) now returns HTML —
we scrape the dataset page directly instead.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

OPENALEX_BASE = "https://api.openalex.org"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
OPENALEX_MAILTO = "sota-finder@example.com"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "sota-finder/1.0 (research tool; mailto:sota-finder@example.com)"
})


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: int = 15,
         retries: int = 3, backoff: float = 5.0) -> Any:
    """Fetch JSON. Returns None on any error."""
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                print(f"  [Rate limited] waiting {wait:.0f}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "json" not in ct:
                return None
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP {e.response.status_code}] {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"  [Network error] {url}: {e}")
            return None
    return None


def _get_html(url: str, timeout: int = 15) -> str | None:
    """Fetch raw HTML. Returns None on any error."""
    try:
        r = _SESSION.get(url, timeout=timeout)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "")
            if "html" in ct or "text" in ct:
                return r.text
        return None
    except requests.exceptions.RequestException:
        return None


def _get_text(url: str, params: dict | None = None, timeout: int = 15) -> str | None:
    """Fetch raw text/XML. Returns None on any error."""
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.text
        return None
    except requests.exceptions.RequestException:
        return None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            words.append((pos, word))
    return " ".join(w for _, w in sorted(words))


def _is_acronym(name: str) -> bool:
    alpha = re.sub(r"[^A-Za-z]", "", name)
    return (len(alpha) >= 2 and alpha == alpha.upper()
            and len(name) <= 12 and " " not in name)


def _mentions_dataset(record: dict, dataset_name: str, full_name: str = "") -> bool:
    """True if title or abstract contains the dataset name OR full name."""
    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    haystack = f"{title} {abstract}"

    # Check acronym / short name
    if _is_acronym(dataset_name):
        if re.search(r"\b" + re.escape(dataset_name) + r"\b", haystack, re.I):
            return True
    else:
        haystack_lower = haystack.lower()
        needle = dataset_name.lower()
        if needle in haystack_lower:
            return True
        needle_norm = needle.replace("-", " ").replace("_", " ")
        if needle_norm in haystack_lower.replace("-", " ").replace("_", " "):
            return True

    # Check full name (if provided) — any 4-word substring is enough
    if full_name and len(full_name) > 10:
        haystack_lower = haystack.lower()
        fn_words = full_name.lower().split()
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in haystack_lower:
                return True

    return False


def _openalex_work_to_dict(work: dict, source: str = "openalex") -> dict:
    title = (work.get("title") or "").strip()
    doi = work.get("doi") or ""
    oa = work.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    loc = work.get("primary_location") or {}
    landing = (loc.get("landing_page_url") or "").strip()
    url = oa_url or landing or doi or "UNKNOWN"
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    return {
        "title": title,
        "year": work.get("publication_year") or "UNKNOWN",
        "paper_url": url,
        "abstract": abstract,
        "source": source,
    }


def _task_terms_from_text(text: str) -> list[str]:
    """Extract broad task/domain terms for benchmark expansion queries."""
    lower = text.lower()
    terms: list[str] = []
    for pattern, term in [
        (r"\beeg|electroencephal", "EEG"),
        (r"\becg|ekg|electrocardio", "ECG"),
        (r"\bsleep|psg|polysomn", "sleep"),
        (r"\bemotion|affective", "emotion recognition"),
        (r"\bwearable|accelerometer|ppg|photopleth", "wearable"),
        (r"\bbci|brain-computer", "BCI"),
    ]:
        if re.search(pattern, lower):
            terms.append(term)
    return list(dict.fromkeys(terms))


# ---------------------------------------------------------------------------
# Step 1: Identify the original dataset paper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Modality keyword map
# Maps UI label → search/scoring keywords (all lowercase)
# ---------------------------------------------------------------------------

MODALITY_KEYWORDS: dict[str, list[str]] = {
    "EEG":                         ["eeg", "electroencephalograph", "brain-computer interface", "bci", "eeg signal"],
    "ECG / EKG":                   ["ecg", "ekg", "electrocardiograph", "cardiac", "arrhythmia", "heart rate variability", "hrv"],
    "fMRI":                        ["fmri", "functional mri", "bold", "neuroimaging", "brain imaging"],
    "ECoG":                        ["ecog", "electrocorticograph", "intracranial eeg"],
    "EMG":                         ["emg", "electromyograph", "muscle"],
    "PPG / BVP":                   ["ppg", "bvp", "photoplethysmograph", "pulse oximetry"],
    "EDA / GSR":                   ["eda", "gsr", "galvanic skin", "electrodermal", "skin conductance"],
    "Wearable / IMU / Accelerometer": ["wearable", "imu", "accelerometer", "gyroscope", "inertial measurement"],
    "PSG (polysomnography)":       ["psg", "polysomnograph", "sleep staging", "sleep study"],
    "Eye tracking / EOG":          ["eog", "eye tracking", "electrooculograph", "gaze"],
    "fNIRS":                       ["fnirs", "near-infrared spectroscopy", "nirs"],
    "Audio / speech":              ["speech", "audio", "acoustic", "voice"],
    "Video / facial":              ["facial expression", "video", "face recognition", "rgb"],
    "Multimodal":                  ["multimodal", "multi-modal", "fusion"],
}


def _modality_keywords_flat(modalities: list[str]) -> list[str]:
    """Return all keyword strings for the given UI modality labels."""
    keywords: list[str] = []
    for m in modalities:
        keywords.extend(MODALITY_KEYWORDS.get(m, [m.lower().split("/")[0].strip()]))
    return list(dict.fromkeys(keywords))  # deduplicated, order-preserving


def _has_modality_signal(text: str, keywords: list[str]) -> bool:
    """True if any modality keyword appears in the text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _has_dataset_name_signal(text: str, dataset_name: str, full_name: str = "") -> bool:
    """True if text explicitly mentions the dataset acronym/short name or full name."""
    if not text:
        return False

    if _is_acronym(dataset_name):
        # Prefer exact-case acronym evidence. Case-insensitive matching turns
        # common-word dataset names like DREAMT/FACED into many false positives,
        # so lowercase mentions only count when they appear near dataset context.
        if re.search(r"\b" + re.escape(dataset_name) + r"\b", text):
            return True
        lower = text.lower()
        name_lower = dataset_name.lower()
        for m in re.finditer(r"\b" + re.escape(name_lower) + r"\b", lower):
            window = lower[max(0, m.start() - 90): m.end() + 90]
            if re.search(
                r"\b(dataset|database|benchmark|corpus|recordings?|"
                r"eeg|ecg|ekg|ppg|psg|wearable|physionet|electroenceph|"
                r"electrocardio|sleep|emotion|affective)\b",
                window,
            ):
                return True
    else:
        haystack_lower = text.lower()
        needle = dataset_name.lower()
        if needle in haystack_lower:
            return True
        needle_norm = needle.replace("-", " ").replace("_", " ")
        if needle_norm in haystack_lower.replace("-", " ").replace("_", " "):
            return True

    if full_name and len(full_name) > 10:
        haystack_lower = text.lower()
        fn_words = full_name.lower().split()
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in haystack_lower:
                return True

    return False


def _expanded_dataset_queries(dataset_name: str) -> list[str]:
    """Return conservative biomedical query expansions for dataset discovery.

    The expansions avoid case-specific title shortcuts. They cover common
    physiology acronym patterns such as pECG/pEEG, where the published dataset
    paper may spell out "pediatric electrocardiogram" instead of the dataset
    short name.
    """
    queries = [dataset_name]
    compact = re.sub(r"[^A-Za-z0-9]", "", dataset_name)
    lower = compact.lower()
    if "pecg" in lower:
        queries.extend([
            "pediatric ECG database",
            "pediatric electrocardiogram database",
            "children ECG database disease diagnosis",
        ])
    if "peeg" in lower:
        queries.extend([
            "pediatric EEG database",
            "pediatric electroencephalogram database",
        ])
    if "ecg" in lower and "pecg" not in lower:
        queries.append("ECG database")
    if "eeg" in lower and "peeg" not in lower:
        queries.append("EEG dataset")
    if _is_acronym(dataset_name):
        queries.extend([
            f"{dataset_name} EEG dataset",
            f"{dataset_name} physiological dataset",
            f"{dataset_name} emotion recognition dataset",
            f"{dataset_name} affective computing dataset",
        ])
    return list(dict.fromkeys(q for q in queries if q.strip()))


def _looks_like_dataset_record(title: str, text: str, dataset_name: str, full_name: str = "") -> bool:
    """Guard repository hits so common words do not outrank real papers."""
    combined = f"{title} {text}"
    if not _has_dataset_name_signal(combined, dataset_name, full_name):
        return False
    lower = combined.lower()
    return bool(re.search(
        r"\b(dataset|database|benchmark|corpus|recordings?|signals?|"
        r"physionet|zenodo|figshare|openneuro|eeg|ecg|ekg|ppg|psg|"
        r"wearable|electroenceph|electrocardio|sleep|emotion|affective|"
        r"participants?|subjects?|patients?)\b",
        lower,
    ))


def _has_biomedical_signal(text: str) -> bool:
    return bool(re.search(
        r"\b(eeg|ecg|ekg|ppg|psg|emg|eog|wearable|sleep|emotion|affective|"
        r"physiolog|biomed|clinical|patient|participants?|subjects?|"
        r"electroenceph|electrocardio|cardiovascular|pediatric|paediatric|"
        r"disease|diagnosis|brain|neuro)\b",
        text,
        re.I,
    ))


_OA_DATASET_PAPER_SELECT = (
    "id,title,cited_by_count,publication_year,doi,"
    "authorships,primary_location,abstract_inverted_index"
)

# Common English verbs/adjectives that would match single-word acronym datasets
_COMMON_VERB_PATTERNS = re.compile(
    r"\b(was|were|is|are|had|have|has|being|been|get|got|become|feel|felt)\s+{name}\b",
    re.I,
)


def _score_as_dataset_paper(
    work: dict,
    dataset_name: str,
    full_name: str = "",
    modality_keywords: list[str] | None = None,
) -> float:
    """Score how likely this OpenAlex work is the paper that *introduced* the dataset.

    Higher = more likely to be the defining paper, not a paper that uses it.
    Pass full_name for a strong title-match bonus.
    Pass modality_keywords to penalise papers from the wrong domain.
    """
    import math

    title = (work.get("title") or "").lower()
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index")).lower()
    name = dataset_name.lower()
    name_upper = dataset_name.upper()
    fn = (full_name or work.get("_full_name") or "").lower()

    score = 0.0

    # --- Full name match (strongest signal when provided) ---
    # The full name in a title almost uniquely identifies the defining paper.
    if fn and len(fn) > 10:
        fn_words = fn.split()
        # Require at least 4 consecutive words from the full name to appear in title
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in title:
                score += 15.0
                break
        # Full name in abstract (paper defines the acronym in the body)
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in abstract:
                score += 8.0
                break

    # --- Strong positive signals (paper defines/introduces the dataset) ---

    # "ACRONYM:" in title — nearly definitive (e.g. "WESAD: Wearable Stress...")
    if re.search(r"\b" + re.escape(name) + r"\s*:", title):
        score += 12.0

    # "ACRONYM dataset" or "ACRONYM benchmark" in title
    if re.search(r"\b" + re.escape(name) + r"\s+(dataset|benchmark|corpus|database|collection)\b", title):
        score += 10.0

    # "the ACRONYM" in title (e.g. "The DREAMT Dataset")
    if re.search(r"\bthe\s+" + re.escape(name) + r"\b", title):
        score += 4.0

    # Abstract introduces the acronym as a name, not a word
    intro_re = re.compile(
        r"(we\s+(introduce|present|release|propose|collect|describe|publish)\s+(the\s+)?"
        + re.escape(name)
        + r"|"
        + re.escape(name)
        + r"\s*[,\s]+a\s+new\s+(dataset|benchmark|corpus|database)"
        + r"|"
        + r"\(" + re.escape(name_upper) + r"\)"   # (DREAMT) — acronym definition syntax
        + r"|"
        + r"called\s+['\"]?" + re.escape(name) + r"['\"]?"
        + r"|"
        + r"named\s+['\"]?" + re.escape(name) + r"['\"]?"
        + r")",
        re.I,
    )
    if intro_re.search(abstract):
        score += 8.0

    # "dataset" or "benchmark" appears near the name in abstract (within 60 chars)
    for m in re.finditer(re.escape(name), abstract):
        window = abstract[max(0, m.start() - 60): m.end() + 60]
        if re.search(r"\b(dataset|benchmark|corpus|database|collection)\b", window):
            score += 3.0
            break

    # --- Strong positive signals: content only found in dataset-introducing papers ---
    # Data collection / recruitment section markers
    if re.search(r"\b(data\s+collection|participants?|recruitment|we\s+recruited|we\s+collected"
                 r"|recording\s+protocol|annotation\s+process|data\s+availability"
                 r"|irb\s+approval|institutional\s+review\s+board|ethics\s+approval"
                 r"|informed\s+consent|publicly\s+available\s+at|can\s+be\s+downloaded)\b",
                 abstract, re.I):
        score += 5.0

    # Demographics / equipment detail — only in papers describing actual data collection
    if re.search(r"\b(age|gender|bmi|height|weight)\s*(=|:|range|distribution|mean|±)\b"
                 r"|\b(amplifier|electrode|sensor|sampling\s+rate|acquisition)\b",
                 abstract, re.I):
        score += 3.0

    # --- Negative signals: method/model papers that merely USE a dataset ---

    # Common verb/copula patterns: "was faced", "have dreamt", "were named"
    if re.search(r"\b(was|were|is|are|had|have|has|been|felt)\s+" + re.escape(name) + r"\b", title):
        score -= 12.0

    # Gerund / participle: "facing", "dreaming" — not relevant but penalise wrong root
    if re.search(r"\b" + re.escape(name) + r"(ing|ed|s)\b", title, re.I) and not re.search(
        r"\b" + re.escape(name) + r"\b", title, re.I
    ):
        score -= 5.0

    # Method-paper title signals: "novel", "proposed", "outperform", "state-of-the-art"
    if re.search(r"\b(novel\s+(method|approach|framework|model|algorithm)"
                 r"|proposed\s+(method|approach|model)"
                 r"|benchmark\s+study"
                 r"|deep\s+learning\s+methods?"
                 r"|classification\b"
                 r"|outperform(s|ing)?\s"
                 r"|state[\s\-]of[\s\-]the[\s\-]art)\b", title, re.I):
        score -= 8.0

    # Method-paper abstract signals: "we propose", "our approach achieves", "we evaluate on"
    if re.search(r"\b(we\s+propose\s+a?\s*(novel|new)?"
                 r"|our\s+(approach|method|model)\s+achieves"
                 r"|we\s+evaluate\s+(our|on|the)\b"
                 r"|we\s+demonstrate\s+that\s+our)\b", abstract, re.I):
        score -= 6.0

    # Title contains ONLY the dataset name as a common word (no dataset/benchmark signal)
    if name in title and not re.search(
        r"\b(dataset|benchmark|corpus|database|EEG|ECG|PSG|fMRI|wearable|sleep|ECoG|EMG|PPG|physiolog)\b",
        title, re.I,
    ):
        score -= 2.0

    if re.search(r"\b(dataset|database|corpus|collection)\b", title, re.I) and _has_biomedical_signal(title):
        score += 8.0
    source_name = (((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "").lower()
    if "scientific data" in source_name:
        score += 4.0
    if re.search(r"\bdisease\s+diagnos", title, re.I):
        score += 3.0
    if re.search(r"\bcovering\s+\d|participants?|subjects?|patients?|children\b", title, re.I):
        score += 2.0

    if not _has_biomedical_signal(f"{title} {abstract}"):
        score -= 15.0

    # --- Modality bonus (hard filter applied separately after scoring) ---
    if modality_keywords:
        if _has_modality_signal(f"{title} {abstract}", modality_keywords):
            score += 6.0

    has_dataset_signal = _has_dataset_name_signal(f"{title} {abstract}", dataset_name, full_name)

    has_dataset_signal = _has_dataset_name_signal(f"{title} {abstract}", dataset_name, full_name)

    # --- Signal 3: repository DOI prefix — near-certain dataset source ---
    doi = (work.get("doi") or "").lower()
    _REPO_DOIS = {
        "10.13026": 20.0,   # PhysioNet
        "10.5281/zenodo": 18.0,  # Zenodo
        "10.6084/m9.figshare": 18.0,  # Figshare
        "10.18112": 16.0,   # NSRR (National Sleep Research Resource)
        "openneuro.org": 16.0,
        "huggingface.co/datasets": 14.0,
        "10.7910": 14.0,    # Harvard Dataverse
        "10.17632": 14.0,   # Mendeley Data
        "10.5061/dryad": 14.0,  # Dryad
        "kaggle.com/datasets": 12.0,
    }
    for prefix, bonus in _REPO_DOIS.items():
        if prefix in doi:
            score += bonus if has_dataset_signal else 2.0
            break

    # Also check if title/abstract mentions hosting platforms by name
    _PLATFORM_TERMS = ["physionet", "zenodo", "openneuro", "huggingface", "figshare",
                       "nsrr", "national sleep research", "harvard dataverse", "mendeley data"]
    if any(p in f"{title} {abstract}" for p in _PLATFORM_TERMS):
        score += 5.0 if has_dataset_signal else 1.0

    # --- Citation count bonus (log scale, capped) ---
    citations = work.get("cited_by_count") or 0
    if citations > 0:
        score += min(math.log10(citations + 1) * 1.5, 4.0)

    # Recency penalty: very new papers rarely have had time to become seminal
    year = work.get("publication_year") or 2024
    if isinstance(year, int) and year >= 2025:
        score -= 2.0

    return score


def _oa_search(params: dict) -> list[dict]:
    """OpenAlex search helper — returns list of works or []."""
    data = _get(f"{OPENALEX_BASE}/works", params=params)
    return (data or {}).get("results") or []


def _s2_candidates(dataset_name: str, modality_context: str = "") -> list[dict]:
    """Semantic Scholar semantic search for the dataset paper.

    S2's relevance ranking handles acronym disambiguation better than
    keyword search because it uses semantic context, not just term frequency.
    modality_context: space-separated keywords (e.g. "EEG electroencephalography")
    to steer results toward the correct domain.
    """
    context = modality_context or "EEG ECG physiological wearable biomedical"
    params = {
        "query": f"{dataset_name} dataset introduced benchmark {context}",
        "fields": "paperId,title,citationCount,abstract,year,externalIds,authors,venue",
        "limit": 8,
    }
    data = _get(f"{S2_BASE}/paper/search", params=params, retries=2, backoff=3.0)
    return (data or {}).get("data") or []


def _work_url(work: dict) -> str:
    """Extract the best available URL from an OpenAlex/fake-OA work dict."""
    oa = work.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    loc = work.get("primary_location") or {}
    landing = (loc.get("landing_page_url") or "").strip()
    doi_raw = work.get("doi") or ""
    doi_link = (
        f"https://doi.org/{doi_raw.lstrip('https://doi.org/')}"
        if doi_raw and doi_raw not in ("UNKNOWN", "") else ""
    )
    return oa_url or landing or doi_link or ""


def _extract_dataset_stats(text: str) -> dict:
    """Extract dataset statistics from plain text (abstract or HTML body).

    Returns a dict with best-effort values; missing fields are empty strings.
    """
    t = text  # preserve case for label extraction; use lower() per-pattern

    def _first(pattern: str, flags: int = re.I) -> str:
        m = re.search(pattern, t, flags)
        return m.group(1).replace(",", "").strip() if m else ""

    # Subjects / participants
    subjects = _first(
        r"(\d[\d,]*)\s*(?:healthy\s+)?(?:subjects?|participants?|patients?"
        r"|volunteers?|individuals?|healthy\s+adults?|neonates?|infants?)"
    )

    # EEG/ECG channels or leads
    channels = _first(
        r"(\d+)[- ]?(?:channel|electrode|lead|sensor)s?\b"
        r"|(\d+)[- ]?ch\b"
    )
    if not channels:
        # try reversed form: "channel count of N"
        channels = _first(r"(?:channel|electrode|lead)\s+count\s+of\s+(\d+)")

    # Total recording duration
    duration = ""
    dur_h = _first(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?\b|h\b)")
    dur_m = _first(r"(\d+(?:\.\d+)?)\s*(?:minutes?|min\b)")
    if dur_h:
        duration = f"{dur_h} h"
        if dur_m:
            duration += f" {dur_m} min"
    elif dur_m:
        duration = f"{dur_m} min"

    # Dataset size
    size = _first(r"(\d+(?:\.\d+)?)\s*(?:GB|MB|TB|gigabytes?|megabytes?|terabytes?)")
    size_unit_m = re.search(
        r"(\d+(?:\.\d+)?)\s*(GB|MB|TB|gigabytes?|megabytes?|terabytes?)", t, re.I
    )
    if size_unit_m:
        size = f"{size_unit_m.group(1)} {size_unit_m.group(2).upper().rstrip('S').replace('IGABYTE','B').replace('EGABYTE','B').replace('ERABYTE','B')}"

    # Files / recordings / trials
    files = _first(
        r"(\d[\d,]*)\s*(?:files?|recordings?|segments?|clips?|trials?\b)"
        r"|(?:total\s+of\s+)(\d[\d,]*)\s*(?:files?|recordings?)"
    )

    # Class labels — look for enumerated lists near label/class/stage/category keywords
    labels: list[str] = []
    # Pattern: "N classes: A, B, C" or "labels: A, B, C" or "stages: Wake, N1..."
    label_list_m = re.search(
        r"(?:classes?|labels?|categories?|stages?|conditions?|annotations?)"
        r"\s*(?:include|are|:|=|–|-)\s*([A-Za-z0-9 ,/\-]+?)(?:\.|;|\n|and\s+\w+\s+\()",
        t, re.I,
    )
    if label_list_m:
        raw = label_list_m.group(1)
        parts = [p.strip() for p in re.split(r"[,/]", raw) if p.strip()]
        labels = [p for p in parts if 1 <= len(p) <= 30][:12]

    # Fallback: well-known label sets by keyword
    if not labels:
        if re.search(r"\bsleep\s+stag", t, re.I):
            if re.search(r"\bN3\b|\bN4\b|\bslow.wave\b", t, re.I):
                labels = ["Wake", "N1", "N2", "N3", "REM"]
            elif re.search(r"\b4.class|\bfour.class\b", t, re.I):
                labels = ["Wake", "Light", "Deep", "REM"]
        elif re.search(r"\barrhythmi|\bAFib\b|\batrial\s+fib", t, re.I):
            labels = ["Normal sinus rhythm", "AF", "Other rhythm", "Noisy"]
        elif re.search(r"\bemotion", t, re.I):
            vals = re.findall(r"\b(valence|arousal|dominance|happy|sad|fear|anger|disgust|surprise|neutral)\b", t, re.I)
            labels = list(dict.fromkeys(v.capitalize() for v in vals))[:8]

    # Train / val / test split hint
    split = ""
    split_m = re.search(
        r"(\d+)\s*[/\-:]\s*(\d+)\s*[/\-:]\s*(\d+)\s*"
        r"(?:train(?:ing)?[/\-]val(?:idation)?[/\-]test|split)",
        t, re.I,
    )
    if split_m:
        split = f"{split_m.group(1)}/{split_m.group(2)}/{split_m.group(3)} (train/val/test)"
    else:
        train_m = re.search(r"(\d+)\s*(?:subjects?|participants?)\s*(?:for\s+)?train(?:ing)?", t, re.I)
        test_m  = re.search(r"(\d+)\s*(?:subjects?|participants?)\s*(?:for\s+)?test(?:ing)?", t, re.I)
        if train_m and test_m:
            split = f"{train_m.group(1)} train / {test_m.group(1)} test subjects"

    return {
        "subjects":  subjects,
        "channels":  channels,
        "duration":  duration,
        "size":      size,
        "files":     files,
        "labels":    labels,
        "split":     split,
    }


def find_dataset_paper(
    dataset_name: str,
    full_name: str | None = None,
    modalities: list[str] | None = None,
) -> dict:
    """Find the paper that introduced the dataset.

    Uses a multi-strategy candidate collection followed by scoring to handle:
    - Datasets whose acronym is also a common English word (DREAMT, FACED)
    - Datasets where the acronym only appears in the abstract, not the title
    - Datasets where the full name differs from the acronym

    Pass full_name (e.g. "Dataset for Real-time EEG-based and AMbulatory Testing")
    for much more reliable identification — the full name in a title almost
    uniquely identifies the dataset paper.

    Returns a metadata dict with keys:
      found, title, authors, author_ids, venue, year, doi, openalex_id,
      citation_count, pwc_listed
    """
    _empty: dict = {
        "found": False, "title": "UNKNOWN", "authors": [], "author_ids": [],
        "venue": "UNKNOWN", "year": "UNKNOWN", "doi": "UNKNOWN",
        "paper_url": "UNKNOWN",
        "openalex_id": None, "citation_count": 0, "pwc_listed": False,
        "full_name": full_name or "",
    }

    is_acronym = _is_acronym(dataset_name)
    name = dataset_name.lower()
    dataset_queries = _expanded_dataset_queries(dataset_name)
    mod_keywords = _modality_keywords_flat(modalities) if modalities else None
    _base_params = {
        "mailto": OPENALEX_MAILTO,
        "select": _OA_DATASET_PAPER_SELECT,
        "per-page": "8",
    }

    # ---- Collect candidates from multiple queries ----
    candidates: list[dict] = []
    seen_ids: set[str] = set()

    def _add(works: list[dict]) -> None:
        for w in works:
            wid = w.get("id") or w.get("paperId") or ""
            if wid not in seen_ids:
                seen_ids.add(wid)
                candidates.append(w)

    # Query 0 (highest precision): if full name provided, search by it in the title.
    # The full name in a title almost uniquely identifies the dataset paper.
    if full_name:
        # Use first 5+ words of full name to avoid OpenAlex filter length issues
        fn_words = full_name.strip().split()
        fn_query = " ".join(fn_words[:8])
        _add(_oa_search({**_base_params,
            "filter": f"title.search:{fn_query},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))
        # Also search abstract for the full name (paper may define acronym in abstract)
        _add(_oa_search({**_base_params,
            "filter": f"abstract.search:{fn_query},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))
        # S2 semantic search with full name — much more targeted than acronym alone
        _mod_ctx = " ".join((mod_keywords or [])[:4])
        for s2_work in _s2_candidates(full_name, modality_context=_mod_ctx):
            s2_title = (s2_work.get("title") or "").strip()
            if not s2_title:
                continue
            ext_ids = s2_work.get("externalIds") or {}
            fake_oa: dict = {
                "id": f"s2:{s2_work.get('paperId', '')}",
                "title": s2_title,
                "cited_by_count": s2_work.get("citationCount") or 0,
                "publication_year": s2_work.get("year"),
                "doi": ext_ids.get("DOI"),
                "authorships": [
                    {"author": {"display_name": a.get("name", ""), "id": ""}}
                    for a in (s2_work.get("authors") or [])
                ],
                "primary_location": {"source": {"display_name": s2_work.get("venue") or ""}},
                "_abstract_text": (s2_work.get("abstract") or "").lower(),
            }
            if fake_oa["id"] not in seen_ids:
                seen_ids.add(fake_oa["id"])
                candidates.append(fake_oa)

    # Signal 1 — dataset-introduction verbs: words that appear in papers that
    # *introduce* a dataset, not in papers that merely *use* one.
    for query in dataset_queries:
        for intro_phrase in (
            f"{query} dataset",
            f"introduce {query}",
            f"present {query}",
            f"release {query}",
            f"collected {query}",
            f"recruited {query}",
            f"data collection {query}",
            f"publicly available {query}",
        ):
            _add(_oa_search({**_base_params,
                "filter": f"title.search:{intro_phrase},publication_year:>2010",
                "sort": "cited_by_count:desc",
            }))
            _add(_oa_search({**_base_params,
                "filter": f"abstract.search:{intro_phrase},publication_year:>2010",
                "sort": "cited_by_count:desc",
            }))
            if len(candidates) >= 30:
                break
        if len(candidates) >= 30:
            break

    # Signal 2 — hosting-platform searches: combine dataset name with the platforms
    # that host biomedical/ML datasets. A DOI from one of these is a near-certain match.
    for platform in ("PhysioNet", "Zenodo", "HuggingFace", "OpenNeuro", "NSRR",
                     "Physionet", "figshare", "Kaggle"):
        _add(_oa_search({**_base_params,
            "filter": f"abstract.search:{dataset_name} {platform},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))

    # Title search (NAME: / NAME dataset/benchmark pattern)
    for query in dataset_queries:
        _add(_oa_search({**_base_params,
            "filter": f"title.search:{query},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))

    # PubMed/PMC indexes many Scientific Data and biomedical dataset papers
    # earlier and more reliably than broad OpenAlex title filters.
    for query in dataset_queries:
        pubmed_search = _get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": f"{query} dataset database",
                "retmode": "json",
                "retmax": 5,
                "sort": "relevance",
            },
            timeout=10,
            retries=1,
            backoff=2.0,
        )
        ids = ((pubmed_search or {}).get("esearchresult") or {}).get("idlist") or []
        if not ids:
            continue
        pubmed_summary = _get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=10,
            retries=1,
            backoff=2.0,
        )
        result = (pubmed_summary or {}).get("result") or {}
        for uid in ids:
            item = result.get(uid) or {}
            title_p = item.get("title") or ""
            if not title_p:
                continue
            fake_pm: dict = {
                "id": f"pubmed:{uid}",
                "title": title_p,
                "cited_by_count": 0,
                "publication_year": str(item.get("pubdate") or "")[:4] or None,
                "doi": next(
                    (a.get("value") for a in (item.get("articleids") or [])
                     if a.get("idtype") == "doi"),
                    None,
                ),
                "authorships": [
                    {"author": {"display_name": a.get("name", ""), "id": ""}}
                    for a in (item.get("authors") or [])
                ],
                "primary_location": {
                    "landing_page_url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                    "source": {"display_name": item.get("fulljournalname") or "PubMed"},
                },
                "_abstract_text": title_p.lower(),
                "_full_name": full_name or "",
            }
            if fake_pm["id"] not in seen_ids:
                seen_ids.add(fake_pm["id"])
                candidates.append(fake_pm)

    # Acronym definition syntax: "(ACRONYM)" in abstract
    if is_acronym:
        _add(_oa_search({**_base_params,
            "filter": f"abstract.search:({dataset_name.upper()}),publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))

    # Semantic Scholar semantic search — handles disambiguation via contextual ranking
    _mod_ctx_acronym = " ".join((mod_keywords or [])[:4])
    for s2_work in _s2_candidates(dataset_name, modality_context=_mod_ctx_acronym):
        s2_title = (s2_work.get("title") or "").strip()
        if not s2_title:
            continue
        ext_ids = s2_work.get("externalIds") or {}
        fake_oa: dict = {
            "id": f"s2:{s2_work.get('paperId', '')}",
            "title": s2_title,
            "cited_by_count": s2_work.get("citationCount") or 0,
            "publication_year": s2_work.get("year"),
            "doi": ext_ids.get("DOI"),
            "authorships": [
                {"author": {"display_name": a.get("name", ""), "id": ""}}
                for a in (s2_work.get("authors") or [])
            ],
            "primary_location": {"source": {"display_name": s2_work.get("venue") or ""}},
            "_abstract_text": (s2_work.get("abstract") or "").lower(),
            "_full_name": full_name or "",
        }
        if fake_oa["id"] not in seen_ids:
            seen_ids.add(fake_oa["id"])
            candidates.append(fake_oa)

    # Fallback: search repositories directly
    # These return dataset records rather than paper records, so we wrap them
    # as fake-OA dicts with a repository DOI that gives a huge scoring bonus.
    _repo_searches = [
        # Zenodo dataset search (JSON API)
        (f"https://zenodo.org/api/records",
         {"q": dataset_name, "type": "dataset", "size": 5},
         "zenodo"),
        # PhysioNet search (HTML scrape handled by fetch_physionet already,
        # but we add a direct OpenAlex filter for physionet.org works)
        (None, None, None),
    ]

    # Zenodo direct search
    # PhysioNet direct search is a high-precision source for physiology
    # dataset records and avoids treating common English acronyms as generic
    # web/repository hits.
    for query in dataset_queries:
        slug_guess = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        for slug_p in (slug_guess, slug_guess.replace("-", "")):
            page_url = f"https://physionet.org/content/{slug_p}/"
            page_html = _get_html(page_url, timeout=8) or ""
            if not page_html or len(page_html) < 1000:
                continue
            title_m = re.search(r"<h1[^>]*>(.*?)</h1>", page_html, re.I | re.S)
            title_p = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_m.group(1))).strip() if title_m else query
            if not _looks_like_dataset_record(title_p, page_html, dataset_name, full_name or ""):
                continue
            doi_m = re.search(r"\b10\.13026/[A-Za-z0-9.\-]+", page_html)
            fake_pn: dict = {
                "id": f"physionet:{slug_p}",
                "title": title_p,
                "cited_by_count": 0,
                "publication_year": None,
                "doi": doi_m.group(0) if doi_m else "10.13026/physionet",
                "authorships": [],
                "primary_location": {
                    "landing_page_url": page_url,
                    "source": {"display_name": "PhysioNet"},
                },
                "_abstract_text": page_html[:20_000].lower(),
                "_full_name": full_name or "",
            }
            if fake_pn["id"] not in seen_ids:
                seen_ids.add(fake_pn["id"])
                candidates.append(fake_pn)

    physionet_data = _get(
        "https://physionet.org/api/v1/contents/",
        params={"search": dataset_name, "format": "json"},
        timeout=10,
        retries=1,
        backoff=2.0,
    )
    physionet_items = (
        physionet_data if isinstance(physionet_data, list)
        else (physionet_data or {}).get("results", [])
    )
    for item in physionet_items[:5]:
        title_p = item.get("title") or item.get("name") or item.get("slug") or ""
        slug_p = item.get("slug") or item.get("id") or ""
        if not title_p:
            continue
        page_html = _get_html(f"https://physionet.org/content/{slug_p}/", timeout=8) or ""
        if not _looks_like_dataset_record(title_p, page_html, dataset_name, full_name or ""):
            continue
        fake_pn: dict = {
            "id": f"physionet:{slug_p}",
            "title": title_p,
            "cited_by_count": item.get("citation_count") or 0,
            "publication_year": item.get("publication_year") or item.get("year"),
            "doi": item.get("doi") or "10.13026/physionet",
            "authorships": [],
            "primary_location": {
                "landing_page_url": f"https://physionet.org/content/{slug_p}/",
                "source": {"display_name": "PhysioNet"},
            },
            "_abstract_text": page_html[:20_000].lower(),
            "_full_name": full_name or "",
        }
        if fake_pn["id"] not in seen_ids:
            seen_ids.add(fake_pn["id"])
            candidates.append(fake_pn)

    zenodo_data = _get("https://zenodo.org/api/records",
                       params={"q": dataset_name, "type": "dataset", "size": 5},
                       timeout=10, retries=1, backoff=2.0)
    for hit in ((zenodo_data or {}).get("hits", {}).get("hits") or []):
        meta = hit.get("metadata") or {}
        title_z = meta.get("title") or ""
        if not title_z:
            continue
        doi_z = hit.get("doi") or f"10.5281/zenodo.{hit.get('id', '')}"
        description = meta.get("description") or ""
        if not _looks_like_dataset_record(title_z, description, dataset_name, full_name or ""):
            continue
        creators = meta.get("creators") or []
        fake: dict = {
            "id": f"zenodo:{hit.get('id', '')}",
            "title": title_z,
            "cited_by_count": 0,
            "publication_year": (meta.get("publication_date") or "")[:4] or None,
            "doi": doi_z,
            "authorships": [
                {"author": {"display_name": c.get("name", ""), "id": ""}}
                for c in creators
            ],
            "primary_location": {"source": {"display_name": "Zenodo"}},
            "_abstract_text": description.lower(),
            "_full_name": full_name or "",
        }
        if fake["id"] not in seen_ids:
            seen_ids.add(fake["id"])
            candidates.append(fake)

    # HuggingFace datasets API
    hf_data = _get("https://huggingface.co/api/datasets",
                   params={"search": dataset_name, "limit": 5},
                   timeout=10, retries=1, backoff=2.0)
    for ds in (hf_data or []):
        ds_id = ds.get("id") or ""
        if not ds_id:
            continue
        description = ds.get("description") or ""
        title_hf = ds_id.split("/")[-1].replace("-", " ").replace("_", " ")
        if not _looks_like_dataset_record(title_hf, description, dataset_name, full_name or ""):
            continue
        fake_hf: dict = {
            "id": f"hf:{ds_id}",
            "title": title_hf,
            "cited_by_count": ds.get("downloads") or 0,
            "publication_year": None,
            "doi": f"huggingface.co/datasets/{ds_id}",
            "authorships": [{"author": {"display_name": ds_id.split("/")[0], "id": ""}}],
            "primary_location": {"source": {"display_name": "HuggingFace Datasets"}},
            "_abstract_text": description.lower(),
            "_full_name": full_name or "",
        }
        if fake_hf["id"] not in seen_ids:
            seen_ids.add(fake_hf["id"])
            candidates.append(fake_hf)

    # If a dataset landing page says "ACRONYM - descriptive dataset title",
    # use the descriptive part to find the actual dataset paper. Repository
    # pages are useful evidence, but scholarly citation chains need a paper
    # anchor whenever one exists.
    descriptor_queries: list[str] = []
    for c in list(candidates):
        title_c = c.get("title") or ""
        stripped = re.sub(r"\b" + re.escape(dataset_name) + r"\b", " ", title_c, flags=re.I)
        stripped = re.sub(r"^[\s:;,\-–—]+|[\s:;,\-–—]+$", "", stripped)
        if len(stripped.split()) >= 3 and _has_biomedical_signal(stripped):
            descriptor_queries.append(stripped)
    for desc_query in list(dict.fromkeys(descriptor_queries))[:5]:
        _add(_oa_search({**_base_params,
            "search": desc_query,
            "sort": "relevance_score:desc",
        }))
        _add(_oa_search({**_base_params,
            "search": f"{dataset_name} {desc_query}",
            "sort": "relevance_score:desc",
        }))
        _add(_oa_search({**_base_params,
            "filter": f"title.search:{desc_query},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))
        _add(_oa_search({**_base_params,
            "filter": f"abstract.search:{desc_query},publication_year:>2010",
            "sort": "cited_by_count:desc",
        }))

    if not candidates:
        return _empty

    # ---- Score every candidate ----
    def _patched_score(work: dict) -> float:
        if "_abstract_text" in work:
            w = dict(work)
            w["_abstract_text_lower"] = w["_abstract_text"]
            return _score_as_dataset_paper_with_abstract(
                w, dataset_name, full_name=full_name or "", modality_keywords=mod_keywords
            )
        return _score_as_dataset_paper(
            work, dataset_name, full_name=full_name or "", modality_keywords=mod_keywords
        )

    scored_pairs = sorted(
        [(c, _patched_score(c)) for c in candidates], key=lambda x: x[1], reverse=True
    )

    # ---- Hard filter: all three identifiers must appear in paper/page ----
    # Priority tiers (descending):
    #   Tier 1: acronym + full_name + modality all present
    #   Tier 2: acronym + full_name (modality only in full paper body)
    #   Tier 3: acronym + modality
    #   Tier 4: acronym only (fallback)
    #   Tier 5: top-scored regardless (last resort)

    def _work_title(w: dict) -> str:
        return (w.get("title") or "").lower()

    def _work_text(w: dict) -> str:
        abstract_txt = (
            _reconstruct_abstract(w.get("abstract_inverted_index")).lower()
            or w.get("_abstract_text", "").lower()
        )
        return f"{_work_title(w)} {abstract_txt}"

    def _anywhere_has_acronym(w: dict, extra_html: str = "") -> bool:
        combined = _work_text(w) + " " + extra_html[:15_000].lower()
        return _has_dataset_name_signal(combined, dataset_name)

    def _title_has_fullname(w: dict) -> bool:
        if not full_name or len(full_name) <= 10:
            return True  # not provided — treat as satisfied
        fn_words_chk = full_name.lower().split()
        title = _work_title(w)
        for i in range(max(1, len(fn_words_chk) - 3)):
            if " ".join(fn_words_chk[i:i + 4]) in title:
                return True
        return False

    def _anywhere_has_fullname_v2(w: dict, extra_html: str = "") -> bool:
        if not full_name or len(full_name) <= 10:
            return True
        combined = _work_text(w) + " " + extra_html[:15_000].lower()
        return _has_dataset_name_signal(combined, "", full_name)

    def _anywhere_has_modality(w: dict, extra_html: str = "") -> bool:
        if not mod_keywords:
            return True
        combined = _work_text(w) + " " + extra_html[:15_000].lower()
        return _has_modality_signal(combined, mod_keywords)

    def _anywhere_has_expanded_dataset_query(w: dict, extra_html: str = "") -> bool:
        combined = _work_text(w) + " " + extra_html[:15_000].lower()
        for query in dataset_queries:
            if query == dataset_name:
                continue
            query_lower = query.lower()
            if query_lower in combined and _has_biomedical_signal(combined):
                return True
        return False

    # Cache fetched HTML to avoid re-fetching the same URL
    _html_cache: dict[str, str] = {}

    def _html_for(w: dict) -> str:
        url = _work_url(w)
        if not url:
            return ""
        if url not in _html_cache:
            _html_cache[url] = _get_html(url, timeout=10) or ""
        return _html_cache[url]

    best = None
    missing_flags: set[str] = set()

    # Tiers — acronym and full name must be in the TITLE.
    # Modality may be anywhere (title, abstract, or fetched HTML body).
    tier_tests = []
    if full_name and mod_keywords:
        tier_tests.append((
            "anywhere:acronym+fullname + modality",
            lambda w, html: _anywhere_has_acronym(w, html)
            and _anywhere_has_fullname_v2(w, html)
            and _anywhere_has_modality(w, html),
        ))
    if full_name:
        tier_tests.append((
            "anywhere:acronym+fullname",
            lambda w, html: _anywhere_has_acronym(w, html)
            and _anywhere_has_fullname_v2(w, html),
        ))
    if mod_keywords:
        tier_tests.append((
            "anywhere:acronym + modality",
            lambda w, html: _anywhere_has_acronym(w, html)
            and _anywhere_has_modality(w, html),
        ))
    tier_tests.extend([
        ("expanded biomedical dataset query",
         lambda w, html: _anywhere_has_expanded_dataset_query(w, html)),
        ("anywhere:acronym only",
         lambda w, html: _anywhere_has_acronym(w, html)),
    ])

    for tier_label, tier_test in tier_tests:
        for w, _ in scored_pairs:
            # Quick check without fetching HTML first
            if tier_test(w, ""):
                best = w
            else:
                html = _html_for(w)
                if tier_test(w, html):
                    best = w
            if best is w:
                html_cached = _html_cache.get(_work_url(w), "")
                if not _anywhere_has_acronym(w, html_cached):
                    missing_flags.add("acronym")
                if not _anywhere_has_fullname_v2(w, html_cached):
                    missing_flags.add("full_name")
                if not _anywhere_has_modality(w, html_cached):
                    missing_flags.add("modality")
                print(f"  Dataset paper selected (tier: {tier_label}): \"{(w.get('title') or '')[:65]}\"")
                break
        if best:
            break

    if not best:
        best = scored_pairs[0][0]
        missing_flags = {"acronym", "full_name", "modality"}
        print("  ⚠️  No candidate passed any filter — using top-scored as fallback")

    modality_verified = "modality" not in missing_flags

    # ---- Extract metadata from chosen best ----
    authorships = best.get("authorships") or []
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in authorships if a.get("author", {}).get("display_name")
    ]
    author_ids = [
        (a.get("author", {}).get("id") or "").replace("https://openalex.org/", "")
        for a in authorships
    ]
    loc = best.get("primary_location") or {}
    venue = (loc.get("source") or {}).get("display_name") or "UNKNOWN"
    work_id = (best.get("id") or "").replace("https://openalex.org/", "")
    if work_id.startswith("s2:"):
        work_id = None

    # ---- Extract paper URL ----
    oa = best.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    loc2 = best.get("primary_location") or {}
    landing = (loc2.get("landing_page_url") or "").strip()
    doi_raw = best.get("doi") or ""
    doi_link = (f"https://doi.org/{doi_raw.lstrip('https://doi.org/')}"
                if doi_raw and doi_raw != "UNKNOWN" else "")
    paper_url = oa_url or landing or doi_link or "UNKNOWN"

    # ---- Check PapersWithCode ----
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    pwc_html = _get_html(f"https://paperswithcode.com/dataset/{slug}", timeout=8)
    pwc_listed = bool(pwc_html and len(pwc_html) > 500 and "Page Not Found" not in pwc_html[:2000])
    if pwc_listed and paper_url == "UNKNOWN":
        paper_url = f"https://paperswithcode.com/dataset/{slug}"

    # ---- Extract dataset statistics ----
    # Build a text corpus from every source we have:
    # abstract, already-fetched HTML (from filter step), PapersWithCode page
    stats_text_parts: list[str] = [_work_text(best)]
    # Reuse any HTML already fetched for the selected candidate (no extra request)
    if paper_url and paper_url != "UNKNOWN":
        cached = _html_cache.get(paper_url, "")
        if not cached:
            cached = _get_html(paper_url, timeout=10) or ""
        stats_text_parts.append(cached[:40_000])
    if pwc_html:
        stats_text_parts.append(pwc_html[:20_000])
    stats = _extract_dataset_stats(" ".join(stats_text_parts))

    _empty.update({
        "found": True,
        "title": best.get("title") or "UNKNOWN",
        "authors": authors,
        "author_ids": [aid for aid in author_ids if aid],
        "venue": venue,
        "year": best.get("publication_year") or "UNKNOWN",
        "doi": best.get("doi") or "UNKNOWN",
        "paper_url": paper_url,
        "openalex_id": work_id,
        "citation_count": best.get("cited_by_count") or 0,
        "pwc_listed": pwc_listed,
        "modality_verified": modality_verified,
        "missing_flags": missing_flags,
        "stats": stats,
        "_pwc_html": pwc_html,
        "_candidate_ranking": [
            {
                "title": c.get("title") or "UNKNOWN",
                "score": score,
                "source": (((c.get("primary_location") or {}).get("source") or {}).get("display_name"))
                          or str(c.get("id", "")).split(":", 1)[0]
                          or "UNKNOWN",
            }
            for c, score in scored_pairs[:10]
        ],
    })
    return _empty


def _score_as_dataset_paper_with_abstract(
    work: dict,
    dataset_name: str,
    full_name: str = "",
    modality_keywords: list[str] | None = None,
) -> float:
    """Variant of _score_as_dataset_paper that uses pre-stored abstract text
    (for Semantic Scholar candidates that don't have an inverted index)."""
    import math

    title = (work.get("title") or "").lower()
    abstract = work.get("_abstract_text_lower") or work.get("_abstract_text") or ""
    name = dataset_name.lower()
    name_upper = dataset_name.upper()
    fn = (full_name or work.get("_full_name") or "").lower()
    score = 0.0
    has_dataset_signal = _has_dataset_name_signal(f"{title} {abstract}", dataset_name, full_name)

    # Full name match bonus
    if fn and len(fn) > 10:
        fn_words = fn.split()
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in title:
                score += 15.0
                break
        for i in range(len(fn_words) - 3):
            phrase = " ".join(fn_words[i:i + 4])
            if phrase in abstract:
                score += 8.0
                break

    if re.search(r"\b" + re.escape(name) + r"\s*:", title):
        score += 12.0
    if re.search(r"\b" + re.escape(name) + r"\s+(dataset|benchmark|corpus|database|collection)\b", title):
        score += 10.0
    if re.search(r"\bthe\s+" + re.escape(name) + r"\b", title):
        score += 4.0

    intro_re = re.compile(
        r"(we\s+(introduce|present|release|propose|collect|describe|publish)\s+(the\s+)?"
        + re.escape(name)
        + r"|\(" + re.escape(name_upper) + r"\)"
        + r"|called\s+['\"]?" + re.escape(name) + r"['\"]?"
        + r"|named\s+['\"]?" + re.escape(name) + r"['\"]?"
        + r")", re.I,
    )
    if intro_re.search(abstract):
        score += 8.0

    for m in re.finditer(re.escape(name), abstract):
        window = abstract[max(0, m.start() - 60): m.end() + 60]
        if re.search(r"\b(dataset|benchmark|corpus|database|collection)\b", window):
            score += 3.0
            break

    if re.search(r"\b(was|were|is|are|had|have|has|been|felt)\s+" + re.escape(name) + r"\b", title):
        score -= 12.0

    # Strong positive: dataset-paper section markers / data-collection language
    if re.search(r"\b(data\s+collection|participants?|recruitment|we\s+recruited|we\s+collected"
                 r"|recording\s+protocol|annotation\s+process|data\s+availability"
                 r"|irb\s+approval|institutional\s+review\s+board|ethics\s+approval"
                 r"|informed\s+consent|publicly\s+available\s+at|can\s+be\s+downloaded)\b",
                 abstract, re.I):
        score += 5.0

    if re.search(r"\b(age|gender|bmi|height|weight)\s*(=|:|range|distribution|mean|±)\b"
                 r"|\b(amplifier|electrode|sensor|sampling\s+rate|acquisition)\b",
                 abstract, re.I):
        score += 3.0

    # Strong negative: method-paper signals
    if re.search(r"\b(novel\s+(method|approach|framework|model|algorithm)"
                 r"|proposed\s+(method|approach|model)"
                 r"|benchmark\s+study"
                 r"|deep\s+learning\s+methods?"
                 r"|classification\b"
                 r"|outperform(s|ing)?\s"
                 r"|state[\s\-]of[\s\-]the[\s\-]art)\b", title, re.I):
        score -= 8.0

    if re.search(r"\b(we\s+propose\s+a?\s*(novel|new)?"
                 r"|our\s+(approach|method|model)\s+achieves"
                 r"|we\s+evaluate\s+(our|on|the)\b"
                 r"|we\s+demonstrate\s+that\s+our)\b", abstract, re.I):
        score -= 6.0

    if not _has_biomedical_signal(f"{title} {abstract}"):
        score -= 15.0

    if re.search(r"\b(dataset|database|corpus|collection)\b", title, re.I) and _has_biomedical_signal(title):
        score += 8.0
    source_name = (((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "").lower()
    if "scientific data" in source_name:
        score += 4.0
    if re.search(r"\bdisease\s+diagnos", title, re.I):
        score += 3.0
    if re.search(r"\bcovering\s+\d|participants?|subjects?|patients?|children\b", title, re.I):
        score += 2.0

    # --- Signal 3: repository DOI prefix ---
    doi = (work.get("doi") or "").lower()
    _REPO_DOIS = {
        "10.13026": 20.0, "10.5281/zenodo": 18.0, "10.6084/m9.figshare": 18.0,
        "10.18112": 16.0, "openneuro.org": 16.0, "huggingface.co/datasets": 14.0,
        "10.7910": 14.0, "10.17632": 14.0, "10.5061/dryad": 14.0,
        "kaggle.com/datasets": 12.0,
    }
    for prefix, bonus in _REPO_DOIS.items():
        if prefix in doi:
            score += bonus if has_dataset_signal else 2.0
            break
    _PLATFORM_TERMS = ["physionet", "zenodo", "openneuro", "huggingface", "figshare",
                       "nsrr", "national sleep research", "harvard dataverse", "mendeley data"]
    if any(p in f"{title} {abstract}" for p in _PLATFORM_TERMS):
        score += 5.0 if has_dataset_signal else 1.0

    # --- Modality bonus (hard filter applied separately after scoring) ---
    if modality_keywords:
        if _has_modality_signal(f"{title} {abstract}", modality_keywords):
            score += 6.0

    citations = work.get("cited_by_count") or 0
    if citations > 0:
        score += min(math.log10(citations + 1) * 1.5, 4.0)

    year = work.get("publication_year") or 2024
    if isinstance(year, int) and year >= 2025:
        score -= 2.0

    return score


# ---------------------------------------------------------------------------
# Step 2b: Follow the research group — same authors' follow-up work
# ---------------------------------------------------------------------------

def fetch_by_dataset_authors(
    dataset_name: str,
    author_ids: list[str],
    min_year: int = 2018,
) -> list[dict]:
    """Find follow-up papers from the same group that published the dataset.

    Searches by OpenAlex author ID — more precise than name matching.
    Uses first author and last author (PI) since they co-authored the dataset.
    """
    if not author_ids:
        return []

    # Use first and last author (first = primary contributor, last = PI)
    targets = list(dict.fromkeys(
        [author_ids[0]] + ([author_ids[-1]] if len(author_ids) > 1 else [])
    ))

    results: list[dict] = []
    seen: set[str] = set()

    for author_id in targets:
        params = {
            "filter": (
                f"authorships.author.id:{author_id},"
                f"publication_year:>{min_year - 1}"
            ),
            "per-page": 25,
            "mailto": OPENALEX_MAILTO,
            "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
            "sort": "publication_date:desc",
        }
        data = _get(f"{OPENALEX_BASE}/works", params=params)
        if not data or not data.get("results"):
            continue
        for work in data["results"]:
            title = (work.get("title") or "").strip()
            if not title or title.lower() in seen:
                continue
            r = _openalex_work_to_dict(work, source="research_group")
            if _mentions_dataset(r, dataset_name):
                seen.add(title.lower())
                results.append(r)

    if results:
        print(f"  [Research Group] {len(results)} paper(s) from dataset authors mentioning dataset")
    return results


# ---------------------------------------------------------------------------
# Step 3a: PMLR / CHIL / MICCAI proceedings via OpenAlex venue filter
# ---------------------------------------------------------------------------

# OpenAlex source IDs for key biomedical ML venues
_PMLR_FILTER = (
    "primary_location.source.display_name.search:Machine+Learning+Research"
)
_MICCAI_FILTER = (
    "primary_location.source.display_name.search:MICCAI"
)


def fetch_pmlr_chil(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Search PMLR (CHIL, AISTATS, ICML) and MICCAI via OpenAlex venue filter."""
    print(f"  [PMLR/CHIL] Searching conference proceedings for: {dataset_name!r}")
    results: list[dict] = []
    seen: set[str] = set()

    for venue_filter in (_PMLR_FILTER, _MICCAI_FILTER):
        params = {
            "filter": (
                f"abstract.search:{dataset_name},"
                f"{venue_filter},"
                f"publication_year:>{min_year - 1}"
            ),
            "per-page": 25,
            "mailto": OPENALEX_MAILTO,
            "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
            "sort": "publication_date:desc",
        }
        data = _get(f"{OPENALEX_BASE}/works", params=params)
        if not data or not data.get("results"):
            continue
        for work in data["results"]:
            title = (work.get("title") or "").strip()
            if not title or title.lower() in seen:
                continue
            seen.add(title.lower())
            results.append(_openalex_work_to_dict(work, source="pmlr_chil"))

    if results:
        print(f"  [PMLR/CHIL] {len(results)} paper(s) in PMLR/MICCAI proceedings")
    else:
        print(f"  [PMLR/CHIL] No results in PMLR/MICCAI proceedings")
    return results


# ---------------------------------------------------------------------------
# Step 3b: OpenReview (NeurIPS, ICLR, ICML workshop papers)
# ---------------------------------------------------------------------------

def fetch_openreview(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Search OpenReview for papers mentioning the dataset."""
    print(f"  [OpenReview] Searching for: {dataset_name!r}")
    params = {
        "term": dataset_name,
        "source": "forum",
        "limit": "25",
        "offset": "0",
    }
    data = _get("https://api.openreview.net/notes/search", params=params, retries=2, backoff=3.0)
    if not data or not data.get("notes"):
        print(f"  [OpenReview] No results")
        return []

    results: list[dict] = []
    for note in data["notes"]:
        content = note.get("content") or {}
        title = (content.get("title") or "").strip()
        if not title:
            continue
        abstract = (content.get("abstract") or "").strip()
        # Timestamps are ms since epoch — convert to year
        tcdate = note.get("tcdate") or note.get("cdate") or 0
        try:
            year = int(str(tcdate)[:4]) if tcdate > 1e9 else int(str(tcdate)[:4])
        except (TypeError, ValueError):
            year = "UNKNOWN"
        if isinstance(year, int) and year < min_year:
            continue
        forum_id = note.get("forum") or note.get("id") or ""
        url = f"https://openreview.net/forum?id={forum_id}" if forum_id else "UNKNOWN"
        record = {"title": title, "year": year, "paper_url": url,
                  "abstract": abstract, "source": "openreview"}
        if _mentions_dataset(record, dataset_name):
            results.append(record)

    print(f"  [OpenReview] {len(results)} paper(s) found")
    return results


# ---------------------------------------------------------------------------
# Step 3c: PubMed for clinical datasets
# ---------------------------------------------------------------------------

_PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PUBMED_TOOL = "sota-finder"
_PUBMED_EMAIL = "sota-finder@example.com"


def fetch_pubmed(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Search PubMed for papers using this dataset (useful for clinical datasets)."""
    print(f"  [PubMed] Searching for: {dataset_name!r}")

    search_params = {
        "db": "pubmed",
        "term": f'"{dataset_name}"[tiab]',
        "datetype": "pdat",
        "mindate": str(min_year),
        "retmax": "20",
        "retmode": "json",
        "tool": _PUBMED_TOOL,
        "email": _PUBMED_EMAIL,
    }
    search_data = _get(f"{_PUBMED_BASE}/esearch.fcgi", params=search_params, retries=2)
    if not search_data:
        print(f"  [PubMed] No response")
        return []

    ids = (search_data.get("esearchresult") or {}).get("idlist") or []
    if not ids:
        print(f"  [PubMed] No results")
        return []

    summary_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json",
        "tool": _PUBMED_TOOL,
        "email": _PUBMED_EMAIL,
    }
    summary_data = _get(f"{_PUBMED_BASE}/esummary.fcgi", params=summary_params, retries=2)
    if not summary_data:
        return []

    uids = (summary_data.get("result") or {}).get("uids") or []
    results: list[dict] = []
    for uid in uids:
        paper = summary_data["result"].get(uid) or {}
        title = (paper.get("title") or "").strip().rstrip(".")
        if not title:
            continue
        year = str(paper.get("pubdate") or "")[:4]
        try:
            if int(year) < min_year:
                continue
        except (TypeError, ValueError):
            pass
        doi = next(
            (link["value"] for link in (paper.get("articleids") or [])
             if link.get("idtype") == "doi"),
            None,
        )
        url = f"https://doi.org/{doi}" if doi else f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
        results.append({"title": title, "year": year, "paper_url": url,
                        "abstract": "", "source": "pubmed"})

    print(f"  [PubMed] {len(results)} paper(s) found")
    return results


# ---------------------------------------------------------------------------
# Strategy 1: OpenAlex — find the dataset paper, fetch all papers citing it
# ---------------------------------------------------------------------------

def fetch_arxiv_benchmark_expansion(
    dataset_name: str,
    context_text: str = "",
    min_year: int = 2024,
) -> list[dict]:
    """Find recent broad benchmark/model papers for the dataset's modality/task.

    Some SOTA-style papers evaluate many public datasets but list individual
    dataset names only in tables or appendices, so title/abstract verification
    by dataset acronym alone misses them. This branch uses conservative
    modality/task terms inferred from the dataset paper, then keeps only recent
    papers that look like benchmark or foundation-model evaluations.
    """
    terms = _task_terms_from_text(f"{dataset_name} {context_text}")
    if not terms:
        return []

    queries: list[str] = []
    if "EEG" in terms:
        queries.append("all:EEG AND all:brain AND all:foundation AND all:model")
        queries.append("all:EEG AND all:benchmark AND all:decoding")
    if "ECG" in terms:
        queries.append("all:ECG AND all:benchmark AND all:classification")
    if "sleep" in terms:
        queries.append("all:sleep AND all:wearable AND all:benchmark")
    if "emotion recognition" in terms and "EEG" in terms:
        queries.append("all:EEG AND all:emotion AND all:recognition")

    if not queries:
        return []

    print(f"  [arXiv expansion] Searching recent benchmark/model papers for: {', '.join(terms)}")
    results: list[dict] = []
    seen: set[str] = set()
    ns = {"a": "http://www.w3.org/2005/Atom"}

    for query in list(dict.fromkeys(queries)):
        xml = _get_text(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": 0,
                "max_results": 12,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=20,
        )
        if not xml:
            continue
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            continue

        for entry in root.findall("a:entry", ns):
            title_el = entry.find("a:title", ns)
            summary_el = entry.find("a:summary", ns)
            id_el = entry.find("a:id", ns)
            published_el = entry.find("a:published", ns)
            title = re.sub(r"\s+", " ", (title_el.text or "")).strip() if title_el is not None else ""
            abstract = re.sub(r"\s+", " ", (summary_el.text or "")).strip() if summary_el is not None else ""
            paper_url = (id_el.text or "").replace("http://", "https://") if id_el is not None else "UNKNOWN"
            if not title or title.lower() in seen:
                continue
            year = "UNKNOWN"
            if published_el is not None and published_el.text:
                year_m = re.match(r"(\d{4})", published_el.text)
                year = int(year_m.group(1)) if year_m else "UNKNOWN"
            if isinstance(year, int) and year < min_year:
                continue

            haystack = f"{title} {abstract}"
            if not _has_modality_signal(haystack, [t.lower() for t in terms]):
                continue
            if re.search(r"\b(review|survey|perspective|opinion)\b", title, re.I):
                continue
            if not re.search(
                r"\b(benchmark|foundation\s+model|experiments?\s+(?:on|across)|"
                r"outperform|state[\s-]of[\s-]the[\s-]art|classification|decoding)\b",
                haystack,
                re.I,
            ):
                continue

            seen.add(title.lower())
            results.append({
                "title": title,
                "year": year,
                "paper_url": paper_url,
                "abstract": abstract,
                "source": "arxiv_expansion",
            })

    print(f"  [arXiv expansion] {len(results)} paper(s) found")
    return results


def _find_dataset_paper_openalex(dataset_name: str) -> str | None:
    """Return the OpenAlex work ID of the most likely seminal dataset paper."""
    # Search for a paper whose title contains the dataset name, sorted by citation count
    params = {
        "filter": f"title.search:{dataset_name}",
        "per-page": 8,
        "mailto": OPENALEX_MAILTO,
        "select": "id,title,cited_by_count,publication_year",
        "sort": "cited_by_count:desc",
    }
    data = _get(f"{OPENALEX_BASE}/works", params=params)
    if not data or not data.get("results"):
        return None

    needle = dataset_name.lower()
    for work in data["results"]:
        title = (work.get("title") or "").lower()
        if needle in title or needle.replace("-", " ") in title:
            work_id = work.get("id", "")
            return work_id.replace("https://openalex.org/", "")

    # Fall back to most-cited result (likely the seminal paper)
    first_id = data["results"][0].get("id", "")
    return first_id.replace("https://openalex.org/", "") if first_id else None


def fetch_openalex_citations(
    dataset_name: str,
    min_year: int = 2018,
    openalex_id: str | None = None,
) -> list[dict]:
    """Find the dataset's seminal paper on OpenAlex, return all papers citing it.

    Pass openalex_id (from find_dataset_paper) to skip the lookup step.
    This catches papers that mention the dataset only in the Methods section.
    """
    work_id = openalex_id or _find_dataset_paper_openalex(dataset_name)
    if not openalex_id:
        print(f"  [OpenAlex Citations] Finding seminal paper for: {dataset_name!r}")
    if not work_id:
        print(f"  [OpenAlex Citations] Could not identify the dataset paper — skipping")
        return []

    print(f"  [OpenAlex Citations] Found dataset paper: {work_id}. Fetching citations…")
    params = {
        "filter": f"cites:{work_id},publication_year:>{min_year - 1}",
        "per-page": 100,
        "mailto": OPENALEX_MAILTO,
        "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
        "sort": "publication_date:desc",
    }
    data = _get(f"{OPENALEX_BASE}/works", params=params)
    if not data or not data.get("results"):
        print(f"  [OpenAlex Citations] No citing papers found")
        return []

    # CS/EE concept IDs — filter to relevant domain
    _CS_EE = {"C41008148", "C202444307", "C119857082", "C154945302"}

    results: list[dict] = []
    seen: set[str] = set()
    for work in data["results"]:
        title = (work.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        # Only keep CS/EE papers (biomedical ML lives here)
        concepts = [c.get("id", "") for c in (work.get("concepts") or [])]
        concept_ids = {c.replace("https://openalex.org/", "") for c in concepts}
        if _CS_EE and not (_CS_EE & concept_ids):
            # Concepts not always present — don't filter if concepts missing
            pass
        seen.add(title.lower())
        results.append(_openalex_work_to_dict(work, source="openalex_citations"))

    print(f"  [OpenAlex Citations] {len(results)} citing paper(s) found")
    return results


# ---------------------------------------------------------------------------
# Strategy 2: Semantic Scholar — citation-based
# ---------------------------------------------------------------------------

def _find_dataset_paper_s2(dataset_name: str) -> str | None:
    """Return the Semantic Scholar paper ID of the most likely seminal dataset paper."""
    params = {
        "query": f'"{dataset_name}" dataset benchmark',
        "fields": "paperId,title,citationCount",
        "limit": 5,
    }
    data = _get(f"{S2_BASE}/paper/search", params=params, retries=2, backoff=3.0)
    if not data or not data.get("data"):
        return None

    needle = dataset_name.lower()
    # Prefer paper whose title contains the dataset name, most-cited
    candidates = sorted(data["data"], key=lambda p: p.get("citationCount") or 0, reverse=True)
    for paper in candidates:
        title = (paper.get("title") or "").lower()
        if needle in title or needle.replace("-", " ") in title:
            return paper.get("paperId")
    return candidates[0].get("paperId") if candidates else None


def fetch_s2_citations(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Find the dataset's seminal paper on Semantic Scholar, return all papers citing it."""
    print(f"  [S2 Citations] Finding seminal paper for: {dataset_name!r}")
    paper_id = _find_dataset_paper_s2(dataset_name)
    if not paper_id:
        print(f"  [S2 Citations] Could not identify the dataset paper — skipping")
        return []

    print(f"  [S2 Citations] Found paper {paper_id}. Fetching citations…")
    params = {
        "fields": "title,year,externalIds,abstract,openAccessPdf",
        "limit": 100,
    }
    data = _get(f"{S2_BASE}/paper/{paper_id}/citations", params=params, retries=2, backoff=3.0)
    if not data or not data.get("data"):
        print(f"  [S2 Citations] No citations found (may be rate-limited)")
        return []

    results: list[dict] = []
    seen: set[str] = set()
    for entry in data["data"]:
        paper = entry.get("citingPaper") or {}
        title = (paper.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        year = paper.get("year") or "UNKNOWN"
        try:
            if int(year) < min_year:
                continue
        except (TypeError, ValueError):
            pass
        seen.add(title.lower())
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi = ext_ids.get("DOI")
        pdf_info = paper.get("openAccessPdf") or {}
        url = (
            pdf_info.get("url")
            or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
            or (f"https://doi.org/{doi}" if doi else None)
            or "UNKNOWN"
        )
        results.append({
            "title": title,
            "year": year,
            "paper_url": url,
            "abstract": (paper.get("abstract") or "").strip(),
            "source": "s2_citations",
        })

    print(f"  [S2 Citations] {len(results)} citing paper(s) found")
    return results


# ---------------------------------------------------------------------------
# Strategy 3 & 4: OpenAlex and Semantic Scholar keyword search (original)
# ---------------------------------------------------------------------------

def fetch_openalex(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Search OpenAlex for papers mentioning the dataset in title or abstract."""
    print(f"  [OpenAlex] Searching for: {dataset_name!r}")
    all_works: list[dict] = []

    for field in ("abstract.search", "title.search"):
        params = {
            "filter": (
                f"{field}:{dataset_name},"
                "concepts.id:C41008148|C202444307,"
                f"publication_year:>{min_year - 1}"
            ),
            "per-page": 30,
            "mailto": OPENALEX_MAILTO,
            "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
            "sort": "publication_date:desc",
        }
        data = _get(f"{OPENALEX_BASE}/works", params=params)
        if data and data.get("results"):
            all_works.extend(data["results"])

    if not all_works:
        params = {
            "search": f"{dataset_name} deep learning classification",
            "per-page": 50,
            "mailto": OPENALEX_MAILTO,
            "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
            "sort": "publication_date:desc",
            "filter": f"concepts.id:C41008148|C202444307,publication_year:>{min_year - 1}",
        }
        data = _get(f"{OPENALEX_BASE}/works", params=params)
        if data and data.get("results"):
            all_works.extend(data["results"])

    if not all_works:
        print(f"  [OpenAlex] No results returned for {dataset_name!r}")
        return []

    seen: set[str] = set()
    raw: list[dict] = []
    for work in all_works:
        title = (work.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        raw.append(_openalex_work_to_dict(work))

    verified = [r for r in raw if _mentions_dataset(r, dataset_name)]
    if not verified and raw:
        print(
            f"  [OpenAlex] ⚠️  {len(raw)} papers fetched but none mention "
            f"'{dataset_name}' in title/abstract. "
            "Try the full dataset name."
        )
        return []

    print(f"  [OpenAlex] {len(verified)} paper(s) verified in title/abstract")
    return verified


def fetch_semantic_scholar(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """Search Semantic Scholar for papers mentioning the dataset."""
    print(f"  [Semantic Scholar] Searching: {dataset_name!r}")
    params = {
        "query": f'"{dataset_name}" dataset',
        "fields": "title,year,externalIds,abstract,openAccessPdf",
        "limit": 30,
    }
    data = _get(f"{S2_BASE}/paper/search", params=params, retries=2, backoff=3.0)
    if not data or not data.get("data"):
        print("  [Semantic Scholar] No results (may be rate-limited)")
        return []

    results: list[dict] = []
    for paper in data["data"]:
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        year = paper.get("year") or "UNKNOWN"
        try:
            if int(year) < min_year:
                continue
        except (TypeError, ValueError):
            pass
        ext_ids = paper.get("externalIds") or {}
        arxiv_id = ext_ids.get("ArXiv")
        doi = ext_ids.get("DOI")
        pdf_info = paper.get("openAccessPdf") or {}
        url = (
            pdf_info.get("url")
            or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None)
            or (f"https://doi.org/{doi}" if doi else None)
            or "UNKNOWN"
        )
        results.append({
            "title": title, "year": year, "paper_url": url,
            "abstract": (paper.get("abstract") or "").strip(),
            "source": "semantic_scholar",
        })

    verified = [r for r in results if _mentions_dataset(r, dataset_name)]
    if not verified and results:
        print(
            f"  [Semantic Scholar] ⚠️  {len(results)} fetched but none mention '{dataset_name}' "
            "in title/abstract."
        )
        return []

    print(f"  [Semantic Scholar] {len(verified)} paper(s) verified")
    return verified


# ---------------------------------------------------------------------------
# Strategy 5: PapersWithCode — scrape the dataset benchmark page
# ---------------------------------------------------------------------------

_PWC_PAPER_RE = re.compile(
    r'href="(/paper/[^"]+)"[^>]*>[\s\S]*?<span[^>]*class="[^"]*item-strip-title[^"]*"[^>]*>([\s\S]*?)</span>',
    re.I,
)
_PWC_TITLE_RE = re.compile(r'<h1[^>]*class="[^"]*paper-title[^"]*"[^>]*>(.*?)</h1>', re.I | re.S)
_PWC_YEAR_RE = re.compile(r'\b(20\d{2})\b')


def _pwc_slug(dataset_name: str) -> str:
    """Convert dataset name to PapersWithCode URL slug."""
    slug = dataset_name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug


def fetch_paperswithcode(
    dataset_name: str,
    min_year: int = 2018,
    _cached_html: str | None = None,
) -> list[dict]:
    """Scrape the PapersWithCode dataset page for listed benchmark papers."""
    slug = _pwc_slug(dataset_name)
    url = f"https://paperswithcode.com/dataset/{slug}"

    html = _cached_html
    if not html:
        print(f"  [PapersWithCode] Fetching: {url}")
        html = _get_html(url)
    else:
        print(f"  [PapersWithCode] Using cached page for: {dataset_name!r}")
    if not html or "Page Not Found" in html[:2000] or len(html) < 500:
        # Try without hyphens between numbers (e.g. "isruc-sleep" vs "isruc-sleep-3")
        alt_slug = re.sub(r"-(\d)", r"\1", slug)
        if alt_slug != slug:
            html = _get_html(f"https://paperswithcode.com/dataset/{alt_slug}")
        if not html or len(html) < 500:
            print(f"  [PapersWithCode] Dataset page not found for {dataset_name!r}")
            return []

    # Extract paper links and titles from benchmark table
    # PapersWithCode renders papers as rows with title + year + metric
    paper_links = re.findall(
        r'href="(https://paperswithcode\.com/paper/[^"]+)"', html
    )
    title_blocks = re.findall(
        r'<a[^>]+href="/paper/[^"]*"[^>]*>\s*([^<]{10,200})\s*</a>', html
    )

    if not paper_links and not title_blocks:
        print(f"  [PapersWithCode] No papers found in page HTML")
        return []

    # Also look for arXiv and DOI links embedded in the page
    arxiv_ids = re.findall(r'arxiv\.org/abs/([\d.]+)', html)
    doi_links = re.findall(r'doi\.org/([^\s"\'<>]+)', html)

    # Build results from title blocks (most reliable on the benchmark table)
    results: list[dict] = []
    seen: set[str] = set()

    for i, title_raw in enumerate(title_blocks[:50]):
        title = re.sub(r'\s+', ' ', title_raw).strip()
        title = re.sub(r'<[^>]+>', '', title).strip()  # strip any remaining tags
        if len(title) < 10 or title.lower() in seen:
            continue
        seen.add(title.lower())

        # Try to find a year near this title block
        idx = html.find(title_raw)
        nearby = html[max(0, idx - 200): idx + 200] if idx != -1 else ""
        year_m = _PWC_YEAR_RE.search(nearby)
        year = int(year_m.group(1)) if year_m else "UNKNOWN"
        if isinstance(year, int) and year < min_year:
            continue

        # Pair with a paper link if available
        paper_url = "UNKNOWN"
        if i < len(paper_links):
            paper_url = paper_links[i]
        elif i < len(arxiv_ids):
            paper_url = f"https://arxiv.org/abs/{arxiv_ids[i]}"

        results.append({
            "title": title, "year": year,
            "paper_url": paper_url, "abstract": "",
            "source": "paperswithcode",
        })

    print(f"  [PapersWithCode] {len(results)} paper(s) found on benchmark page")
    return results


# ---------------------------------------------------------------------------
# Strategy 6: PhysioNet — extract dataset DOI and use for citation lookup
# ---------------------------------------------------------------------------

_PHYSIONET_DOI_RE = re.compile(
    r'(?:doi|DOI)[:\s]*([^\s"\'<>]+/[^\s"\'<>]+)', re.I
)
_PHYSIONET_PAPER_RE = re.compile(
    r'href="(https?://[^"]*(?:doi\.org|arxiv\.org|pubmed|ncbi|proceedings\.mlr\.press)[^"]*)"', re.I
)


def _paper_link_to_record(link: str, source: str) -> dict:
    """Best-effort title extraction for paper links found on dataset pages."""
    title = "[Paper from dataset page - verify URL]"
    abstract = ""
    html = _get_html(link, timeout=10) or ""
    if html:
        h1 = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.I | re.S)
        title_m = h1 or re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if title_m:
            title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title_m.group(1))).strip()
            title = title.replace(" | Proceedings of Machine Learning Research", "")
        abstract_m = re.search(r"<div[^>]+class=\"abstract\"[^>]*>(.*?)</div>", html, re.I | re.S)
        if abstract_m:
            abstract = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", abstract_m.group(1))).strip()
    return {
        "title": title,
        "year": "UNKNOWN",
        "paper_url": link,
        "abstract": abstract,
        "source": source,
    }


def fetch_physionet(dataset_name: str, min_year: int = 2018) -> list[dict]:
    """For PhysioNet datasets: find the dataset page, extract the dataset paper DOI,
    then use OpenAlex to find all papers citing that dataset paper.
    """
    # Try to find the PhysioNet content page
    slug = re.sub(r"[^a-z0-9]+", "-", dataset_name.lower()).strip("-")
    candidates = [
        f"https://physionet.org/content/{slug}/",
        f"https://physionet.org/content/{slug.replace('-', '')}/",
    ]

    html = None
    found_url = None
    for candidate_url in candidates:
        h = _get_html(candidate_url)
        if h and "physionet" in h.lower() and len(h) > 1000:
            html = h
            found_url = candidate_url
            break

    if not html:
        # Try PhysioNet search API
        search_data = _get(
            "https://physionet.org/api/v1/contents/",
            params={"search": dataset_name, "format": "json"},
            retries=1,
        )
        if not search_data:
            print(f"  [PhysioNet] No page found for {dataset_name!r} — skipping")
            return []
        # Find the most relevant result
        items = search_data if isinstance(search_data, list) else search_data.get("results", [])
        if not items:
            return []
        slug = items[0].get("slug") or items[0].get("id") or ""
        html = _get_html(f"https://physionet.org/content/{slug}/")
        if not html:
            return []

    print(f"  [PhysioNet] Found dataset page. Extracting citation DOI…")

    # Extract the dataset paper's DOI from the page
    doi_match = _PHYSIONET_DOI_RE.search(html)
    dataset_doi = doi_match.group(1).strip().rstrip(".") if doi_match else None

    if dataset_doi:
        # Find this paper on OpenAlex by DOI, then get its citations
        doi_data = _get(
            f"{OPENALEX_BASE}/works/doi:{dataset_doi}",
            params={"mailto": OPENALEX_MAILTO, "select": "id,title"},
        )
        if doi_data and doi_data.get("id"):
            work_id = doi_data["id"].replace("https://openalex.org/", "")
            print(f"  [PhysioNet] Found dataset paper on OpenAlex ({work_id}). Fetching citations…")
            params = {
                "filter": f"cites:{work_id},publication_year:>{min_year - 1}",
                "per-page": 100,
                "mailto": OPENALEX_MAILTO,
                "select": "title,publication_year,doi,open_access,abstract_inverted_index,primary_location",
                "sort": "publication_date:desc",
            }
            data = _get(f"{OPENALEX_BASE}/works", params=params)
            if data and data.get("results"):
                results = []
                seen: set[str] = set()
                for work in data["results"]:
                    title = (work.get("title") or "").strip()
                    if not title or title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    results.append(_openalex_work_to_dict(work, source="physionet_citations"))
                for link in _PHYSIONET_PAPER_RE.findall(html):
                    rec = _paper_link_to_record(link, "physionet_page")
                    title = rec["title"]
                    if title and title.lower() not in seen:
                        seen.add(title.lower())
                        results.append(rec)
                print(f"  [PhysioNet] {len(results)} citing paper(s) via DOI citation chain")
                return results

    # Fall back: extract direct paper links from PhysioNet page
    paper_links = _PHYSIONET_PAPER_RE.findall(html)
    if paper_links:
        print(f"  [PhysioNet] Extracted {len(paper_links)} paper link(s) from page")
        results = []
        for link in paper_links[:20]:
            results.append({
                "title": f"[Paper from PhysioNet page — verify URL]",
                "year": "UNKNOWN",
                "paper_url": link,
                "abstract": "",
                "source": "physionet",
            })
        return results

    print(f"  [PhysioNet] Could not extract citations from page")
    return []
