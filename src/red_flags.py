"""SOTA quality filter — red flag detection and paper classification.

Implements the two-tier check system:
  Critical checks 1–6: automatic down-rank if failed
  Secondary checks 7–10: confidence multipliers

Classification:
  STRONG SOTA CANDIDATE  — passes all 6 critical checks
  WEAK SOTA CANDIDATE    — fails 1–2 critical checks
  RELATED WORK ONLY      — fails 3+ critical checks, OR fails check 1/2/4, OR exceeds sanity ceiling

Each flag is stored as "CRITICAL|CHECK_N|message" or "SECONDARY|CHECK_N|message".
"""

from __future__ import annotations

import re

from src.schemas import PaperRecord, UNKNOWN

STRONG = "STRONG SOTA CANDIDATE"
WEAK   = "WEAK SOTA CANDIDATE"
RELATED = "RELATED WORK ONLY"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fulltext(p: PaperRecord) -> str:
    if p.notes and "FullText:" in p.notes:
        return p.notes.split("FullText:", 1)[1]
    return ""

def _abstract(p: PaperRecord) -> str:
    if p.notes and "Abstract:" in p.notes:
        return p.notes.split("Abstract:", 1)[1].split("FullText:", 1)[0]
    return ""

def _alltext(p: PaperRecord) -> str:
    return f"{p.title} {_abstract(p)} {_fulltext(p)}".lower()

def _flag(tier: str, check: str, message: str) -> str:
    return f"{tier}|{check}|{message}"


# ---------------------------------------------------------------------------
# Critical Check 1: Subject-level split
# Absence of explicit confirmation = FAIL.
# ---------------------------------------------------------------------------

_SUBJECT_LEVEL_RE = re.compile(
    r"(subject[\s\-]level\s*(cv|cross[\s\-]?val|split|fold)|"
    r"leave[\s\-]one[\s\-]subject[\s\-]out|loso|"
    r"held[\s\-]out\s+subjects?|"
    r"participant[\s\-]level\s*(split|cv|fold)|"
    r"no\s+subject\s+(in\s+both|overlap|appears\s+in\s+both)|"
    r"cross[\s\-]subject\s+(valid|eval|test)|"
    r"inter[\s\-]subject\s*(valid|eval|test)|"
    r"patient[\s\-]independent|"
    r"subjects?\s+not\s+(seen|present)\s+during\s+train)",
    re.I,
)

_EPOCH_LEVEL_RE = re.compile(
    r"(random(ly)?\s*(split|partition|divid)|"
    r"epoch[\s\-]level\s*split|"
    r"window[\s\-]level\s*split|"
    r"sample[\s\-]level\s*split|"
    r"segment[\s\-]level\s*split|"
    r"\b(80|70|75|90)\s*/\s*(20|30|25|10)\s*(train|split)|"
    r"database\s+(was\s+)?(divided|split)\s+into\s+(train|test))",
    re.I,
)


def _check1_subject_split(p: PaperRecord, text: str) -> str | None:
    if p.split_type == "subject-wise" or _SUBJECT_LEVEL_RE.search(text):
        return None  # PASS
    if p.split_type == "random-window" or _EPOCH_LEVEL_RE.search(text):
        return _flag("CRITICAL", "CHECK_1",
            "Epoch/window-level split confirmed. The model is tested on epochs from subjects "
            "it was trained on — effectively memorising individuals. Scores are inflated and "
            "not comparable to subject-level results.")
    # Absence of subject-level confirmation = FAIL
    return _flag("CRITICAL", "CHECK_1",
        "No explicit subject-level split confirmed. Paper does not state 'subject-level CV', "
        "'held-out subjects', 'LOSO', or equivalent. Assume epoch-level until proven otherwise. "
        "The paper must explicitly confirm subject independence.")


# ---------------------------------------------------------------------------
# Critical Check 2: Cross-validation required for small datasets (<200 subjects)
# ---------------------------------------------------------------------------

_SUBJECT_COUNT_RE = re.compile(
    r"\b(\d+)\s*(subjects?|participants?|patients?|individuals?|volunteers?|recordings?)\b",
    re.I,
)
_CV_CONFIRMED_RE = re.compile(
    r"(k[\s\-]?fold|leave[\s\-]one[\s\-]subject[\s\-]out|loso|"
    r"\d+[\s\-]fold\s*(cross[\s\-])?valid|"
    r"cross[\s\-]valid|repeated\s*random\s*(split|seed)|"
    r"multiple\s*random\s*seed)",
    re.I,
)
_SINGLE_SPLIT_RE = re.compile(
    r"(single\s*(train|test)\s*split|one\s*train[\s\/]test\s*split|"
    r"no\s*cross[\s\-]valid|without\s*cross[\s\-]valid)",
    re.I,
)


def _check2_cross_validation(p: PaperRecord, text: str) -> str | None:
    # Try to extract subject count from text
    counts = [int(m.group(1)) for m in _SUBJECT_COUNT_RE.finditer(text)
              if 5 < int(m.group(1)) < 5000]
    if not counts:
        return None  # Can't assess without knowing dataset size — skip

    min_count = min(counts)
    if min_count >= 200:
        return None  # Large dataset — single split is acceptable

    if _CV_CONFIRMED_RE.search(text):
        return None  # PASS

    if _SINGLE_SPLIT_RE.search(text) or p.cross_validation in (UNKNOWN, "none"):
        return _flag("CRITICAL", "CHECK_2",
            f"Dataset appears to have ~{min_count} subjects (< 200) but no cross-validation "
            "detected. A single train/test split on a small dataset gives an unreliable "
            "estimate of generalisation. Requires k-fold CV (k≥5), LOSO, or multiple seeds.")
    return None


# ---------------------------------------------------------------------------
# Critical Check 3: Metrics have standard deviation
# ---------------------------------------------------------------------------

_STD_RE = re.compile(
    r"(±|\+/-|\bstd\b|s\.d\.|standard\s+deviation|"
    r"confidence\s+interval|\bci\b\s*[:=\(]|"
    r"error\s+bar|variance|\biqr\b|interquartile)",
    re.I,
)


def _check3_std_dev(p: PaperRecord, text: str) -> str | None:
    if not isinstance(p.metric_value, float):
        return None
    search = f"{p.source_snippet} {text[:4000]}"
    if _STD_RE.search(search):
        return None  # PASS
    return _flag("CRITICAL", "CHECK_3",
        f"No standard deviation reported with {p.metric_name} = {p.metric_value:.1f}%. "
        "A single point estimate on a biomedical dataset is unreliable without ± std or "
        "confidence interval across folds or runs.")


# ---------------------------------------------------------------------------
# Critical Check 4: Compares against prior published SOTA on same dataset
# Fails if only self-trained generic baselines are compared.
# ---------------------------------------------------------------------------

_PRIOR_SOTA_RE = re.compile(
    r"(state[\s\-]of[\s\-]the[\s\-]art|previous\s+(best|work|method|result)|"
    r"prior\s+(work|method|result|art)|"
    r"compared\s+(to|with|against)\s+\w+\s+et\s+al|"
    r"\[\d+\].{0,30}(achiev|report|obtain|attain)|"
    r"outperform.{0,50}(published|existing|prior|previous)|"
    r"surpass.{0,50}(published|existing|prior|previous))",
    re.I,
)
_GENERIC_BASELINES_ONLY_RE = re.compile(
    r"(efficientnet|mobilenet|densenet|vgg\d*|alexnet|"
    r"inceptionv?\d*|squeezenet|shufflenet)",
    re.I,
)
_TASK_SPECIFIC_BASELINES_RE = re.compile(
    r"(eegnet|sleep[\s\-]?transformer|u[\s\-]?sleep|seqsleepnet|"
    r"tinysleepnet|sleepfcn|xsleepnet|iitnet|deepsleepnet|"
    r"sleepstager|chambon|phan|supratak|attention[\s\-]?unet|"
    r"resnet[\s\-]?1d|cnn[\s\-]?transformer|mamba|s4|"
    r"li\s+et\s+al|zhang\s+et\s+al|yang\s+et\s+al)",
    re.I,
)


def _check4_prior_sota_comparison(p: PaperRecord, text: str) -> str | None:
    if _PRIOR_SOTA_RE.search(text) or _TASK_SPECIFIC_BASELINES_RE.search(text):
        return None  # PASS
    if not p.models_compared and not _GENERIC_BASELINES_ONLY_RE.search(text):
        return _flag("CRITICAL", "CHECK_4",
            "No comparison against previously published results on this dataset found. "
            "A SOTA claim requires beating the best known prior published result, "
            "not just outperforming generic baselines trained from scratch.")
    if _GENERIC_BASELINES_ONLY_RE.search(text) and not _PRIOR_SOTA_RE.search(text):
        return _flag("CRITICAL", "CHECK_4",
            "Comparisons appear to be against generic image classifiers (EfficientNet, "
            "MobileNet, DenseNet, etc.) trained from scratch — not published SOTA methods "
            "on this dataset. Beating self-trained baselines does not constitute a SOTA claim.")
    return None


# ---------------------------------------------------------------------------
# Critical Check 5: Metric sanity ceiling
# ---------------------------------------------------------------------------

_WEARABLE_MODS = {"ppg", "eda", "bvp", "st", "acc", "gyro", "imu", "wearable"}
_SLEEP_RE = re.compile(r"sleep\s*stag", re.I)


def _check5_metric_sanity(p: PaperRecord, text: str) -> str | None:
    if not isinstance(p.metric_value, float):
        return None
    mv = p.metric_value
    is_wearable_sleep = (
        bool(_SLEEP_RE.search(p.task or ""))
        and bool(_WEARABLE_MODS & {m.lower() for m in p.modalities})
    )
    if is_wearable_sleep:
        metric_lower = (p.metric_name or "").lower()
        if "kappa" in metric_lower and mv > 85.0:
            return _flag("CRITICAL", "CHECK_5",
                f"Kappa = {mv:.2f} on wearable sleep staging exceeds the human inter-rater "
                f"ceiling (~0.75–0.80 kappa). This result is implausible for wrist-worn signals "
                "and likely indicates data leakage or an epoch-level split.")
        if mv > 95.0:
            return _flag("CRITICAL", "CHECK_5",
                f"Accuracy = {mv:.1f}% on wearable sleep staging exceeds the human inter-rater "
                f"ceiling (~80–85%). This result is implausible for wrist-worn signals. "
                "Likely cause: epoch-level split, preprocessing leakage, or overfitting to a "
                "small held-out set.")
    elif mv > 99.0:
        return _flag("CRITICAL", "CHECK_5",
            f"Metric {p.metric_name} = {mv:.1f}% is near-perfect for a biomedical classification "
            "task. This is almost certainly inflated by data leakage or an epoch-level split.")
    elif mv > 97.0:
        return _flag("CRITICAL", "CHECK_5",
            f"Metric {p.metric_name} = {mv:.1f}% is extremely high. Scrutinise split type and "
            "preprocessing pipeline — verify this is a test-set result on held-out subjects.")
    return None


# ---------------------------------------------------------------------------
# Critical Check 6: Data leakage from preprocessing
# ---------------------------------------------------------------------------

_LEAKAGE_BEFORE_RE = re.compile(
    r"(smote|oversamp|undersamp|downsamp|augment|balanced?|"
    r"normaliz[e|ed]|standardiz[e|ed]|z[\s\-]?score|min[\s\-]?max\s+scal)"
    r".{0,100}"
    r"(before\s+(split|partition|divid|train)|"
    r"then\s+(split|partition|divid)|"
    r"prior\s+to\s+(split|partition|train)|"
    r"on\s+the\s+(full|entire|whole)\s+(dataset|data))",
    re.I | re.S,
)
_LEAKAGE_FULL_DATASET_RE = re.compile(
    r"(full|entire|whole|complete)\s+dataset\s+.{0,60}"
    r"(normaliz|standardiz|z[\s\-]?score|balanced?|smote|oversamp)",
    re.I | re.S,
)
_LEAKAGE_CORRECT_RE = re.compile(
    r"(train(ing)?\s+(set|data).{0,40}(normaliz|standardiz|balanced?|smote|oversamp)|"
    r"(normaliz|standardiz|balanced?|smote).{0,40}train(ing)?\s+(set|data)|"
    r"applied\s+(only\s+)?to\s+train|fit\s+on\s+train)",
    re.I,
)


def _check6_data_leakage(p: PaperRecord, text: str) -> str | None:
    if _LEAKAGE_CORRECT_RE.search(text):
        return None  # PASS — explicitly applied to training set only
    if _LEAKAGE_BEFORE_RE.search(text) or _LEAKAGE_FULL_DATASET_RE.search(text):
        return _flag("CRITICAL", "CHECK_6",
            "Possible data leakage: preprocessing (balancing/normalization/augmentation) "
            "may have been applied to the full dataset before the train/test split. "
            "This should be fit on the training set only and applied to test. "
            "Can inflate accuracy by 10–30% on imbalanced datasets.")
    return None


# ---------------------------------------------------------------------------
# Secondary Check 7: Task definition match
# ---------------------------------------------------------------------------

_MULTITASK_RE = re.compile(
    r"(multi[\s\-]?task|auxiliary\s+task|joint(ly)?\s*(train|classif)|"
    r"simultaneously\s+(classif|predict)|"
    r"combined\s+(loss|objective).{0,30}(stage|class|label))",
    re.I,
)
_SUBGROUP_RE = re.compile(
    r"(per[\s\-]group|disorder[\s\-]specific|subgroup\s+accuracy|"
    r"per[\s\-](disorder|condition|class)\s+accuracy\s+as\s+(the\s+)?headline)",
    re.I,
)


def _check7_task_definition(p: PaperRecord, text: str) -> str | None:
    if _MULTITASK_RE.search(text):
        return _flag("SECONDARY", "CHECK_7",
            "Multi-task or auxiliary-task objective detected. The reported metric may combine "
            "or be influenced by an auxiliary loss, making direct comparison to single-task "
            "baselines invalid. Verify what the headline metric represents.")
    if _SUBGROUP_RE.search(text):
        return _flag("SECONDARY", "CHECK_7",
            "Metric may be reported per subgroup or disorder-specific subset. "
            "Verify the headline number covers the full evaluation set and is not "
            "cherry-picked from a favourable subgroup.")
    if p.label_granularity != UNKNOWN:
        return _flag("SECONDARY", "CHECK_7",
            f"Label granularity: {p.label_granularity}. Ensure this matches the field standard "
            "for this dataset before comparing with other papers. Papers using different class "
            "counts cannot be directly ranked against each other.")
    return None


# ---------------------------------------------------------------------------
# Secondary Check 8: Dataset completeness (<80% of subjects used)
# ---------------------------------------------------------------------------

_SUBSET_RE = re.compile(
    r"(we\s+(selected|included|used|chose|retained|excluded)|only|subset\s+of)"
    r".{0,50}\b(\d+)\s*(subjects?|participants?|patients?|individuals?)\b",
    re.I | re.S,
)
_SUBSET_JUSTIFIED_RE = re.compile(
    r"(exclusion\s+criteria|inclusion\s+criteria|artifact\s*(rejection|remov)|"
    r"quality\s+(check|control|threshold)|clinical\s+protocol|"
    r"missing\s+data|insufficient\s+data|poor\s+signal\s+quality)",
    re.I,
)


def _check8_dataset_completeness(p: PaperRecord, text: str) -> str | None:
    m = _SUBSET_RE.search(text)
    if m and not _SUBSET_JUSTIFIED_RE.search(text):
        n = m.group(3)
        return _flag("SECONDARY", "CHECK_8",
            f"Paper uses only ~{n} subjects/participants without a clearly stated clinical "
            "or methodological justification (e.g., artifact exclusion criteria, clinical "
            "protocol). This makes results incomparable to papers using the full dataset. "
            "Cherry-picked subsets can appear to perform better than full-dataset results.")
    return None


# ---------------------------------------------------------------------------
# Secondary Check 9: Code availability (confidence multiplier)
# ---------------------------------------------------------------------------

_CODE_FULL_RE = re.compile(
    r"(code\s+(and\s+)?weights?\s+(are\s+)?(available|released|public)|"
    r"fully?\s+reproducible|model\s+weights?\s+(available|released))",
    re.I,
)
_CODE_PARTIAL_RE = re.compile(
    r"(partial(ly)?\s+(available|released)|"
    r"key\s+components?\s+(of\s+)?code|training\s+code\s+only)",
    re.I,
)
_CODE_ANY_RE = re.compile(
    r"(github\.com|gitlab\.com|zenodo\.org|"
    r"code\s+(is\s+)?(available|released|public|open)|"
    r"open[\s\-]?source|implementation\s+(available|released|public)|"
    r"released\s+(the\s+)?code)",
    re.I,
)
_NO_CODE_VAGUE_RE = re.compile(
    r"(details?\s+(are\s+)?available\s+on\s+request|"
    r"upon\s+(reasonable\s+)?request|"
    r"vague|not\s+(publicly\s+)?available)",
    re.I,
)


def _check9_code_availability(p: PaperRecord, text: str) -> tuple[str | None, float]:
    """Returns (flag_or_None, multiplier)."""
    url = p.paper_url if p.paper_url != UNKNOWN else ""
    has_github = "github" in url.lower() or "github" in text[:2000]

    if _CODE_FULL_RE.search(text) or (has_github and "weight" in text[:3000]):
        return None, 1.0  # Code + weights

    if has_github or _CODE_ANY_RE.search(text):
        return (_flag("SECONDARY", "CHECK_9",
            "Code released but weights/checkpoints not confirmed. "
            "Partial reproducibility — results can be re-trained but not directly verified."),
            0.85)

    if _CODE_PARTIAL_RE.search(text):
        return (_flag("SECONDARY", "CHECK_9",
            "Only partial code released. Full pipeline cannot be reproduced. "
            "Treat unusually high metrics with extra caution."),
            0.70)

    if p.confidence == "high":
        return None, 0.85  # Manually marked high confidence

    if _NO_CODE_VAGUE_RE.search(text):
        return (_flag("SECONDARY", "CHECK_9",
            "No public code — methods are vague or only available on request. "
            "Results cannot be reproduced or independently verified (0.3× confidence multiplier)."),
            0.30)

    return (_flag("SECONDARY", "CHECK_9",
        "No code release found. Detailed methods in paper allow partial reproducibility "
        "but results cannot be verified end-to-end (0.5× confidence multiplier)."),
        0.50)


# ---------------------------------------------------------------------------
# Secondary Check 10: Venue tier (confidence multiplier)
# ---------------------------------------------------------------------------

_TIER1_RE = re.compile(
    r"(nature\s*(medicine|digital\s*medicine|communications)|"
    r"npj\s*digital\s*medicine|"
    r"\bchil\b|pmlr|neurips|nips|icml|iclr|"
    r"ieee\s*tbme|ieee\s*trans.*biomed|"
    r"\bsleep\b\s*(journal|2\d{3})|"
    r"journal\s+of\s+sleep\s+research|"
    r"annals\s+of\s+(the\s+)?amer)",
    re.I,
)
_TIER2_RE = re.compile(
    r"(ieee\s*(j?bhi|journal.*health\s*inform)|"
    r"scientific\s*reports|"
    r"bmc\s*(medical\s*inform|bioinform)|"
    r"\baaai\b|ecml|miccai|embc|"
    r"computers\s+in\s+biology)",
    re.I,
)
_TIER3_RE = re.compile(
    r"(ieee\s*access|"
    r"\bmdpi\b|sensors\s*(mdpi)?|diagnostics\s*(mdpi)?|"
    r"applied\s+sciences\s*(mdpi)?|electronics\s*(mdpi)?|"
    r"smaller\s+ieee\s+conf)",
    re.I,
)
_TIER5_RE = re.compile(r"arxiv\.org", re.I)


def _check10_venue(p: PaperRecord, text: str) -> tuple[str | None, float]:
    """Returns (flag_or_None, multiplier)."""
    url = p.paper_url if p.paper_url != UNKNOWN else ""
    search = f"{url} {text[:1000]}"

    if _TIER5_RE.search(url):
        return (_flag("SECONDARY", "CHECK_10",
            "arXiv preprint — not peer-reviewed. Treat as preliminary results. "
            "(0.3× confidence multiplier)"),
            0.30)
    if _TIER1_RE.search(search):
        return None, 1.0
    if _TIER2_RE.search(search):
        return (_flag("SECONDARY", "CHECK_10",
            "Tier 2 venue (IEEE JBHI, Scientific Reports, MICCAI, etc.). "
            "Good peer review but not top-tier. (0.8× confidence multiplier)"),
            0.80)
    if _TIER3_RE.search(search):
        return (_flag("SECONDARY", "CHECK_10",
            "Tier 3 venue (IEEE Access, MDPI, etc.). "
            "Open-access with faster/lighter review. Scrutinise methodology carefully. "
            "(0.6× confidence multiplier)"),
            0.60)
    # Unknown venue — neutral
    return None, 0.85


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Checks 1 and 4 are individually disqualifying (RELATED WORK regardless of others)
_DISQUALIFYING_CHECKS = {"CHECK_1", "CHECK_2", "CHECK_4"}


def assess_paper(p: PaperRecord) -> tuple[str, float, list[str]]:
    """Run all checks. Returns (sota_category, confidence_score, flags)."""
    text = _alltext(p)

    # ---- Critical checks ----
    critical_flags: list[str] = []
    for check_fn in [
        _check1_subject_split,
        _check2_cross_validation,
        _check3_std_dev,
        _check4_prior_sota_comparison,
        _check5_metric_sanity,
        _check6_data_leakage,
    ]:
        result = check_fn(p, text)
        if result:
            critical_flags.append(result)

    # ---- Secondary checks ----
    secondary_flags: list[str] = []
    code_flag, code_mult = _check9_code_availability(p, text)
    venue_flag, venue_mult = _check10_venue(p, text)

    for check_fn in [_check7_task_definition, _check8_dataset_completeness]:
        result = check_fn(p, text)
        if result:
            secondary_flags.append(result)
    if code_flag:
        secondary_flags.append(code_flag)
    if venue_flag:
        secondary_flags.append(venue_flag)

    all_flags = critical_flags + secondary_flags

    # ---- Classification ----
    failed_codes = {f.split("|")[1] for f in critical_flags}
    n_critical = len(critical_flags)

    if n_critical == 0:
        category = STRONG
    elif n_critical <= 2 and not (failed_codes & _DISQUALIFYING_CHECKS):
        category = WEAK
    else:
        category = RELATED

    # ---- Confidence score ----
    confidence = round(code_mult * venue_mult, 3)

    return category, confidence, all_flags


def apply_red_flags(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Return new PaperRecord list with sota_category, confidence_score, red_flags populated."""
    result = []
    for p in papers:
        category, confidence, flags = assess_paper(p)
        result.append(p.model_copy(update={
            "sota_category": category,
            "confidence_score": confidence,
            "red_flags": flags,
        }))
    return result
