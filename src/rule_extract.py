"""Rule-based field extraction from paper titles and abstracts.

No external API calls — regex + keyword matching only.
Fields that cannot be extracted confidently remain UNKNOWN.
"""

from __future__ import annotations

import re

from src.schemas import PaperRecord, UNKNOWN

# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

_TASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsleep[\s\-]?stag\w*", re.I), "sleep staging"),
    (re.compile(r"\batrial[\s\-]?fibrill\w*|\baf[\s\-]detect\w*", re.I), "atrial fibrillation detection"),
    (re.compile(r"\barrhythmia[\s\-]class\w*|\becg[\s\-]class\w*|\bheart[\s\-]?rhythm[\s\-]class\w*", re.I), "arrhythmia classification"),
    (re.compile(r"\bheartbeat[\s\-]class\w*|\becg[\s\-]beat[\s\-]class\w*", re.I), "heartbeat classification"),
    (re.compile(r"\bseizure[\s\-]detect\w*|\bepilepsy[\s\-]detect\w*|\bictal[\s\-]detect\w*", re.I), "seizure detection"),
    (re.compile(r"\bemotion[\s\-]recogni\w*|\bemotion[\s\-]class\w*|\baffect[\s\-]recogni\w*", re.I), "emotion recognition"),
    (re.compile(r"\bmotor[\s\-]imager\w*|\bmi[\s\-]class\w*", re.I), "motor imagery classification"),
    (re.compile(r"\bmental[\s\-]workload|\bcognitive[\s\-]workload", re.I), "mental workload"),
    (re.compile(r"\bstress[\s\-]detect\w*|\bstress[\s\-]recogni\w*", re.I), "stress detection"),
    (re.compile(r"\bfault[\s\-]detect\w*|\banomaly[\s\-]detect\w*", re.I), "anomaly detection"),
    (re.compile(r"\bbraincomputer[\s\-]interf\w*|\bbci\b", re.I), "BCI"),
    (re.compile(r"\bsignal[\s\-]class\w*|\bphysiolog\w+[\s\-]class\w*", re.I), "signal classification"),
]

# ---------------------------------------------------------------------------
# Metric detection
# ---------------------------------------------------------------------------

_METRIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmacro[\s\-]?f[\s\-]?1\b|\bmf1\b|\bf1[\s\-]?macro\b", re.I), "macro-F1"),
    (re.compile(r"\bweighted[\s\-]?f[\s\-]?1\b|\bf1[\s\-]?weighted\b", re.I), "weighted-F1"),
    (re.compile(r"\bf1[\s\-]score\b|\bf1\b|\bf[\s\-]measure\b|\bf[\s\-]score\b", re.I), "F1"),
    (re.compile(r"\bauc[\s\-]roc\b|\broc[\s\-]auc\b|\bauroc\b|\bauc\b", re.I), "AUC"),
    (re.compile(r"\bcohen['\s]?s?[\s\-]?kappa\b|\bkappa\b|\bcohen[\s\-]kappa\b", re.I), "kappa"),
    (re.compile(r"\bmatthews[\s\-]corr\w*|\bmcc\b", re.I), "MCC"),
    (re.compile(r"\bbalanced[\s\-]accuracy\b", re.I), "balanced accuracy"),
    (re.compile(r"\boverall[\s\-]accuracy\b|\boverall\s+acc\b|\bOA\b", re.I), "overall accuracy"),
    (re.compile(r"\baccuracy\b|\bacc\b", re.I), "accuracy"),
    (re.compile(r"\bsensitivity\b|\brecall\b|\btpr\b", re.I), "sensitivity"),
    (re.compile(r"\bspecificity\b", re.I), "specificity"),
    (re.compile(r"\bprecision\b", re.I), "precision"),
]

# Metric keyword followed (within ~30 chars) by a value
_METRIC_CONTEXT_RE = re.compile(
    r"(?:accuracy|f1|kappa|auc|sensitivity|specificity|precision|recall|mcc|mf1|oa|cohen)"
    r"[^\n]{0,30}?(\d{2,3}(?:\.\d{1,3})?)\s*%"
    r"|(?:accuracy|f1|kappa|auc|sensitivity|specificity|precision|recall|mcc|mf1|oa|cohen)"
    r"[^\n]{0,30}?(0\.\d{2,4})\b",
    re.I,
)

# "achieved/obtained/reached 91.3%" patterns
_ACHIEVE_RE = re.compile(
    r"(?:achiev\w*|obtain\w*|reach\w*|report\w*|yield\w*|attain\w*)"
    r"[^\n]{0,20}?(\d{2,3}(?:\.\d{1,3})?)\s*%",
    re.I,
)

# "of 94.2%" / "= 94.2%" / ": 94.2%"
_OF_EQUALS_RE = re.compile(r"(?:of|=|:)\s*(\d{2,3}(?:\.\d{1,3})?)\s*%", re.I)

# Table-style: "94.2 ± 1.3" or "94.2 ±1.3"
_TABLE_RE = re.compile(r"\b(\d{2,3}(?:\.\d{1,3})?)\s*(?:±|\+/-|\+−)\s*\d+(?:\.\d+)?")

# Bare percentage and decimal
_PERCENT_RE = re.compile(r"\b(\d{1,3}(?:\.\d{1,3})?)\s*%")
_DECIMAL_RE = re.compile(r"\b(0\.\d{2,4})\b")


def _extract_metric(text: str) -> tuple[str, float | str]:
    """Return (metric_name, metric_value). Returns the highest plausible value found."""
    found_metric = UNKNOWN
    for pattern, name in _METRIC_PATTERNS:
        if pattern.search(text):
            found_metric = name
            break

    candidates: list[float] = []

    def _add(raw: str) -> None:
        val = float(raw)
        if 50.0 <= val <= 100.0:
            candidates.append(val)
        elif 0.5 <= val <= 1.0:
            candidates.append(val * 100)

    for m in _METRIC_CONTEXT_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            _add(raw)

    for m in _ACHIEVE_RE.finditer(text):
        _add(m.group(1))

    for m in _OF_EQUALS_RE.finditer(text):
        _add(m.group(1))

    for m in _TABLE_RE.finditer(text):
        _add(m.group(1))

    for raw in _PERCENT_RE.findall(text):
        _add(raw)

    for raw in _DECIMAL_RE.findall(text):
        _add(raw)

    if candidates:
        return found_metric, max(candidates)

    return found_metric, UNKNOWN


# ---------------------------------------------------------------------------
# Model architecture detection
# ---------------------------------------------------------------------------

_MODEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Z][A-Za-z0-9]*Brain\b|\bfoundation\s+model\b", re.I), "Foundation model"),
    (re.compile(r"\btransformer\b", re.I), "Transformer"),
    (re.compile(r"\bbert\b", re.I), "BERT"),
    (re.compile(r"\bvision[\s\-]transformer\b|\bvit\b", re.I), "ViT"),
    (re.compile(r"\bconformer\b", re.I), "Conformer"),
    (re.compile(r"\bmamba\b", re.I), "Mamba"),
    (re.compile(r"\beegnet\b", re.I), "EEGNet"),
    (re.compile(r"\bdeepconvnet\b|\bdeep[\s\-]conv[\s\-]net\b", re.I), "DeepConvNet"),
    (re.compile(r"\bshallowconvnet\b|\bshallow[\s\-]conv[\s\-]net\b", re.I), "ShallowConvNet"),
    (re.compile(r"\bresnet\b|\bres[\s\-]net\b", re.I), "ResNet"),
    (re.compile(r"\binception\w*", re.I), "Inception"),
    (re.compile(r"\blstm\b|\blong[\s\-]short[\s\-]term[\s\-]memory\b", re.I), "LSTM"),
    (re.compile(r"\bgru\b|\bgated[\s\-]recurrent\b", re.I), "GRU"),
    (re.compile(r"\btcn\b|\btemporal[\s\-]convolutional\b", re.I), "TCN"),
    (re.compile(r"\bcnn\b|\bconvolutional[\s\-]neural[\s\-]net\w*", re.I), "CNN"),
    (re.compile(r"\brandom[\s\-]forest\b", re.I), "Random Forest"),
    (re.compile(r"\bsvm\b|\bsupport[\s\-]vector\b", re.I), "SVM"),
    (re.compile(r"\bxgboost\b|\blightgbm\b", re.I), "XGBoost"),
    (re.compile(r"\bautoencoder\b|\bvae\b", re.I), "Autoencoder"),
    (re.compile(r"\bmobilenet\b", re.I), "MobileNet"),
    (re.compile(r"\befficientnet\b", re.I), "EfficientNet"),
    (re.compile(r"\bdeep[\s\-]neural[\s\-]net\w*|\bdnn\b", re.I), "DNN"),
]

# ---------------------------------------------------------------------------
# Split type
# ---------------------------------------------------------------------------

_SUBJECT_WISE_RE = re.compile(
    r"\bsubject[\s\-]?wise\b|\bpatient[\s\-]?wise\b|\bperson[\s\-]wise\b"
    r"|\bleave[\s\-]one[\s\-](?:subject|patient|person)[\s\-]out\b|\bloso\b"
    r"|\bsubject[\s\-]independent\b|\binter[\s\-]subject\b|\bcross[\s\-]subject\b"
    r"|\bpatient[\s\-]independent\b|\binter[\s\-]patient\b|\bcross[\s\-]patient\b"
    r"|\bperson[\s\-]independent\b",
    re.I,
)
_RANDOM_WINDOW_RE = re.compile(
    r"\brandom[\s\-]split\b|\bwindow[\s\-]level\b|\bsample[\s\-]wise\b"
    r"|\bepoch[\s\-]level\b|\brandom[\s\-]window\b|\bsegment[\s\-]level\b"
    r"|\bwithin[\s\-]subject\b|\bintra[\s\-]subject\b"
    r"|\bstratified[\s\-](?:random[\s\-])?split\b",
    re.I,
)


def _extract_split_type(text: str) -> str:
    if _SUBJECT_WISE_RE.search(text):
        return "subject-wise"
    if _RANDOM_WINDOW_RE.search(text):
        return "random-window"
    return UNKNOWN


# ---------------------------------------------------------------------------
# Cross-validation detection
# ---------------------------------------------------------------------------

def _extract_cross_validation(text: str) -> str:
    m = re.search(r"\b(\d+)[\s\-]fold\b", text, re.I)
    if m:
        n = m.group(1)
        if re.search(r"subject|patient|person|speaker", text, re.I):
            return f"{n}-fold subject-wise"
        return f"{n}-fold cross-validation"

    word_num = {
        "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    m = re.search(r"\b(two|three|four|five|six|seven|eight|nine|ten)[\s\-]fold\b", text, re.I)
    if m:
        n = word_num[m.group(1).lower()]
        if re.search(r"subject|patient|person|speaker", text, re.I):
            return f"{n}-fold subject-wise"
        return f"{n}-fold cross-validation"

    if re.search(r"\bloso\b|\bleave[\s\-]one[\s\-](?:subject|patient|person)[\s\-]out\b", text, re.I):
        return "LOSO"

    if re.search(r"\bleave[\s\-]one[\s\-]out\b", text, re.I):
        return "leave-one-out"

    m = re.search(r"\b(\d+)\s*x\s*(?:repeated\s*)?(?:stratified\s*)?cross[\s\-]valid", text, re.I)
    if m:
        return f"{m.group(1)}x cross-validation"

    if re.search(r"\bcross[\s\-]valid\w+\b", text, re.I):
        return "cross-validation"

    return UNKNOWN


# ---------------------------------------------------------------------------
# Modality detection
# ---------------------------------------------------------------------------

_MODALITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\becg\b|\belectrocardiog\w*", re.I), "ECG"),
    (re.compile(r"\beeg\b|\belectroencephalog\w*", re.I), "EEG"),
    (re.compile(r"\bemg\b|\belectromyog\w*", re.I), "EMG"),
    (re.compile(r"\beog\b|\belectrooculog\w*", re.I), "EOG"),
    (re.compile(r"\bppg\b|\bphotoplethysmog\w*", re.I), "PPG"),
    (re.compile(r"\bfmri\b|\bfunctional[\s\-]mri\b|\bbold[\s\-]signal\b", re.I), "fMRI"),
    (re.compile(r"\becog\b|\belectrocorticog\w*", re.I), "ECoG"),
    (re.compile(r"\beda\b|\bgalvanic[\s\-]skin\b|\bgsr\b", re.I), "EDA"),
    (re.compile(r"\baccelerom\w+|\bgyroscop\w+|\bimu\b", re.I), "IMU"),
    (re.compile(r"\brespiration\b|\bbreathing[\s\-]signal\b|\bresp\b", re.I), "respiration"),
    (re.compile(r"\bpsg\b|\bpolysomnogra\w*", re.I), "PSG"),
]

# ---------------------------------------------------------------------------
# Label granularity
# ---------------------------------------------------------------------------

_LABEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b5[\s\-]class\b|\bfive[\s\-]class\b|\b5[\s\-]stage\b|\bW[\s/]N1[\s/]N2[\s/]N3[\s/]REM\b", re.I), "5-class"),
    (re.compile(r"\b4[\s\-]class\b|\bfour[\s\-]class\b|\b4[\s\-]stage\b", re.I), "4-class"),
    (re.compile(r"\b3[\s\-]class\b|\bthree[\s\-]class\b|\b3[\s\-]stage\b", re.I), "3-class"),
    (re.compile(r"\b2[\s\-]class\b|\btwo[\s\-]class\b|\bbinary[\s\-]class\w*|\bnormal[\s\-]vs[\s\-]abnormal\b|\bnormal[\s/.]+abnormal\b", re.I), "binary"),
]

# ---------------------------------------------------------------------------
# Train / val / test split proportions
# ---------------------------------------------------------------------------

def _extract_splits(text: str) -> tuple[str, str, str]:
    """
    Return (train_split, val_split, test_split).
    Each is a percentage string like '80%', a description like 'LOSO', or UNKNOWN.
    """
    # "70/15/15", "80-10-10", "60/20/20 train/val/test"
    m = re.search(r"\b(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{1,2})\b", text)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if abs(a + b + c - 100) <= 2:
            return f"{a}%", f"{b}%", f"{c}%"

    # "80% training … 20% test" (no explicit val)
    m_tr = re.search(r"(\d{1,3})\s*%\s*(?:for\s+)?train\w*", text, re.I)
    m_va = re.search(r"(\d{1,3})\s*%\s*(?:for\s+)?val\w*", text, re.I)
    m_te = re.search(r"(\d{1,3})\s*%\s*(?:for\s+)?test\w*", text, re.I)
    if m_tr or m_te:
        return (
            f"{m_tr.group(1)}%" if m_tr else UNKNOWN,
            f"{m_va.group(1)}%" if m_va else UNKNOWN,
            f"{m_te.group(1)}%" if m_te else UNKNOWN,
        )

    # "train: 70, val: 10, test: 20" or "training set: 80%, test set: 20%"
    m_tr = re.search(r"train\w*\s*[=:]\s*(\d{1,3})\s*%?", text, re.I)
    m_va = re.search(r"val\w*\s*[=:]\s*(\d{1,3})\s*%?", text, re.I)
    m_te = re.search(r"test\w*\s*[=:]\s*(\d{1,3})\s*%?", text, re.I)
    if m_tr or m_te:
        return (
            f"{m_tr.group(1)}%" if m_tr else UNKNOWN,
            f"{m_va.group(1)}%" if m_va else UNKNOWN,
            f"{m_te.group(1)}%" if m_te else UNKNOWN,
        )

    return UNKNOWN, UNKNOWN, UNKNOWN


# ---------------------------------------------------------------------------
# Preprocessing steps (modality-aware)
# ---------------------------------------------------------------------------

def _extract_preprocessing(text: str, modalities: list[str] | None = None) -> list[str]:
    """
    Extract preprocessing steps relevant to the detected modalities.
    Returns a deduplicated, ordered list of step descriptions.
    """
    steps: list[str] = []
    mods = set(modalities or [])

    # ---- Filters --------------------------------------------------------
    # Bandpass range "X-Y Hz" or "between X and Y Hz" or "X to Y Hz"
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*[Hh]z\b"
        r"|between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*[Hh]z\b"
        r"|(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*[Hh]z\b",
        text, re.I,
    )
    if m:
        lo = m.group(1) or m.group(3) or m.group(5)
        hi = m.group(2) or m.group(4) or m.group(6)
        steps.append(f"bandpass {lo}–{hi} Hz")
    else:
        # Separate high-pass / low-pass
        m = re.search(
            r"high[\s\-]?pass\b[^\n]{0,25}?(\d+(?:\.\d+)?)\s*[Hh]z"
            r"|(\d+(?:\.\d+)?)\s*[Hh]z\s+high[\s\-]?pass"
            r"|\bHPF\b[^\n]{0,15}?(\d+(?:\.\d+)?)\s*[Hh]z",
            text, re.I,
        )
        if m:
            val = m.group(1) or m.group(2) or m.group(3)
            steps.append(f"high-pass {val} Hz")

        m = re.search(
            r"low[\s\-]?pass\b[^\n]{0,25}?(\d+(?:\.\d+)?)\s*[Hh]z"
            r"|(\d+(?:\.\d+)?)\s*[Hh]z\s+low[\s\-]?pass"
            r"|\bLPF\b[^\n]{0,15}?(\d+(?:\.\d+)?)\s*[Hh]z",
            text, re.I,
        )
        if m:
            val = m.group(1) or m.group(2) or m.group(3)
            steps.append(f"low-pass {val} Hz")

    # Notch filter
    m = re.search(
        r"notch\s+(?:filter(?:ed)?\s+(?:at\s+)?)?(\d+)\s*[Hh]z"
        r"|(\d+)\s*[Hh]z\s+notch",
        text, re.I,
    )
    if m:
        val = m.group(1) or m.group(2)
        steps.append(f"notch {val} Hz")

    # ---- Sampling rate --------------------------------------------------
    already_in_steps = " ".join(steps)
    m = re.search(
        r"(?:re|down)?sampl\w+\s+(?:rate\s+)?(?:at\s+|to\s+|of\s+)?(\d{2,5})\s*[Hh]z\b"
        r"|(?:sampling|sample)\s+(?:rate|freq\w*)\s*(?:of\s+)?(\d{2,5})\s*[Hh]z\b"
        r"|(\d{2,5})\s*[Hh]z\s+(?:sampling|sample)\b",
        text, re.I,
    )
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        if val and val not in already_in_steps:
            steps.append(f"sampling rate {val} Hz")

    # ---- Window / epoch segmentation ------------------------------------
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*[-\s]?(?:second|sec)\s*(?:epoch|windows?|segments?|trial|sample)\b"
        r"|(?:epoch|window|segment)\s+(?:length|size|duration)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*s\b"
        r"|(\d+(?:\.\d+)?)\s*s\s+(?:epoch|window|segment)\b",
        text, re.I,
    )
    if m:
        val = m.group(1) or m.group(2) or m.group(3)
        steps.append(f"{val}-second window")
    else:
        m = re.search(r"(\d{3,4})\s*[-\s]?sample\s*(?:window|epoch|segment)\b"
                      r"|(\d{3,4})\s+samples?\s+per\s+(?:epoch|window|segment)\b", text, re.I)
        if m:
            val = m.group(1) or m.group(2)
            steps.append(f"{val}-sample window")

    # ---- Signal transforms / frequency-domain features -----------------
    if re.search(r"\bSTFT\b|\bshort[\s\-]time\s+(?:Fourier|FFT)\b|\bspectrogram\b", text, re.I):
        steps.append("STFT/spectrogram")
    if re.search(
        r"\bCWT\b|\bDWT\b|\bcontinuous\s+wavelet\b|\bdiscrete\s+wavelet\b"
        r"|\bwavelet\s+(?:transform|decomposition|features|coefficients)\b",
        text, re.I,
    ):
        steps.append("wavelet transform")
    if re.search(r"\bFFT\b|\bfast\s+Fourier\s+transform\b|\bfrequency[\s\-]domain\b", text, re.I):
        if not any("STFT" in s or "spectrogram" in s for s in steps):
            steps.append("FFT")
    if re.search(
        r"\bband[\s\-]?power\b|\bPSD\b|\bpower\s+spectral\s+density\b|\bWelch\b", text, re.I
    ):
        steps.append("band power/PSD")

    # ---- EEG-specific ---------------------------------------------------
    if not mods or "EEG" in mods or "ECoG" in mods:
        # ICA
        if re.search(r"\bICA\b|\bindependent\s+component\s+anal", text, re.I):
            steps.append("ICA")
        # ASR
        if re.search(r"\bASR\b|\bartifact\s+subspace\s+reconstruct", text, re.I):
            steps.append("ASR")
        elif re.search(r"\bartifact\s+(?:reject|remov|correct)\w*", text, re.I):
            steps.append("artifact rejection")
        # EOG artifact removal
        if re.search(
            r"\bEOG\b[^\n]{0,40}(?:remov|correct|regress|artifact)"
            r"|(?:remov|correct|regress)[^\n]{0,20}\bEOG\b",
            text, re.I,
        ):
            steps.append("EOG removal")
        # Baseline correction
        if re.search(r"\bbaseline\s+(?:correct|remov|subtract)\w*", text, re.I):
            steps.append("baseline correction")
        # CSP
        if re.search(r"\bCSP\b|\bcommon\s+spatial\s+pattern\b", text, re.I):
            steps.append("CSP")
        # Reference electrode
        if re.search(r"average\s+reference\b|\bcommon\s+average\s+ref\w*|\bCAR\b", text, re.I):
            steps.append("average reference")
        elif re.search(r"(?:linked\s+)?mastoid\s+ref\w*", text, re.I):
            steps.append("mastoid reference")
        elif re.search(r"\bCz\s+ref\w*|\bvertex\s+ref\w*", text, re.I):
            steps.append("Cz reference")
        # Channel count
        m = re.search(r"(\d+)[\s\-](?:channel|electrode|EEG\s+channel)", text, re.I)
        if m:
            steps.append(f"{m.group(1)} channels")

    # ---- ECG-specific ---------------------------------------------------
    if not mods or "ECG" in mods or "PPG" in mods:
        if re.search(r"pan[\s\-]tompkins\b|R[\s\-]peak\s+detect\w*", text, re.I):
            steps.append("R-peak detection (Pan-Tompkins)")
        if not any("artifact" in s or "ICA" in s for s in steps):
            if re.search(r"\bartifact\s+(?:reject|remov)\w*", text, re.I):
                steps.append("artifact rejection")

    # ---- fMRI-specific --------------------------------------------------
    if "fMRI" in mods:
        m = re.search(
            r"\bTR\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:s\b|sec\b|ms\b)"
            r"|repetition\s+time\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|ms)\b",
            text, re.I,
        )
        if m:
            steps.append(f"TR = {m.group(1) or m.group(2)}s")
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*mm\s+FWHM\b|FWHM\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mm\b",
            text, re.I,
        )
        if m:
            steps.append(f"{m.group(1) or m.group(2)} mm FWHM smoothing")
        if re.search(r"motion\s+correct\w*|\bhead\s+motion\b", text, re.I):
            steps.append("motion correction")
        if re.search(r"\bGLM\b|\bgeneral\s+linear\s+model\b", text, re.I):
            steps.append("GLM")

    # ---- General --------------------------------------------------------
    if not any("ICA" in s or "artifact" in s for s in steps):
        if re.search(r"\bartifact\s+(?:reject|remov|correct)\w*", text, re.I):
            steps.append("artifact rejection")

    # Named filter types (when no numeric cutoff was captured)
    if not any("bandpass" in s or "high-pass" in s or "low-pass" in s for s in steps):
        if re.search(r"\bButterworth\b", text, re.I):
            steps.append("Butterworth filter")
        elif re.search(r"\bFIR\b|\bIIR\b", text):
            steps.append("FIR/IIR filter")

    # Architecture-level preprocessing
    if re.search(r"\bpatch\s+embed\w*|\bpatch\s+token\w*|\bpatching\b", text, re.I):
        steps.append("patch embedding")
    if re.search(r"\btokeniz\w+\b", text, re.I) and not any("patch" in s for s in steps):
        steps.append("tokenization")

    if re.search(r"\bz[\s\-]?score\s+normali[sz]\w*|\bz[\s\-]?score\b", text, re.I):
        steps.append("z-score normalization")
    elif re.search(r"\bmin[\s\-]?max\s+normali[sz]\w*", text, re.I):
        steps.append("min-max normalization")
    elif re.search(r"\bnormali[sz]\w*|\bstandardiz\w*", text, re.I):
        steps.append("normalization")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CLOSE_VALUE_RE = re.compile(
    r"(?:of|:|=|is|was|were|at|to)\s*(\d{2,3}(?:\.\d{1,3})?)\s*%"
    r"|(?:of|:|=|is|was)\s*(0\.\d{2,4})\b"
    r"|(\d{2,3}(?:\.\d{1,3})?)\s*%",
    re.I,
)


def extract_all_metrics(text: str) -> list[tuple[str, float]]:
    """
    Return all (metric_name, value_%) pairs found in text, sorted by value descending.
    Looks for numbers immediately following each metric keyword (tight window).
    Skips duplicate values already attributed to a higher-priority metric.
    """
    results: dict[str, float] = {}
    claimed_values: set[float] = set()

    for pat, name in _METRIC_PATTERNS:
        best: float | None = None
        for m in pat.finditer(text):
            # Look after the keyword first (e.g. "accuracy of 94.5%")
            after = text[m.end(): min(len(text), m.end() + 50)]
            cm = _CLOSE_VALUE_RE.search(after)
            # Fallback: look before the keyword (e.g. "94.5% accuracy")
            if not cm:
                before = text[max(0, m.start() - 30): m.start()]
                cm = _CLOSE_VALUE_RE.search(before)
            if cm:
                raw = cm.group(1) or cm.group(2) or cm.group(3)
                if raw:
                    val = float(raw)
                    if val <= 1.0:
                        val *= 100
                    if 50.0 <= val <= 100.0:
                        best = max(best or 0.0, val)

        if best is not None and best not in claimed_values:
            results[name] = best
            claimed_values.add(best)

    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def _abstract_from_notes(notes: str) -> str:
    """Return the richest text available: full text if fetched, else abstract."""
    if notes and "FullText:" in notes:
        return notes.split("FullText:", 1)[1].strip()
    if notes and "Abstract:" in notes:
        return notes.split("Abstract:", 1)[1].strip()
    return ""


def enrich_records(papers: list[PaperRecord]) -> list[PaperRecord]:
    """Return new list with rule-extracted fields merged in.
    Only overwrites UNKNOWN — never downgrades an already-known value.
    """
    enriched: list[PaperRecord] = []
    for p in papers:
        abstract = _abstract_from_notes(p.notes)
        full_text = f"{p.title} {abstract}"
        updates: dict = {}

        if p.task == UNKNOWN:
            for pat, name in _TASK_PATTERNS:
                if pat.search(full_text):
                    updates["task"] = name
                    break

        if p.metric_name == UNKNOWN or p.metric_value == UNKNOWN:
            m_name, m_val = _extract_metric(full_text)
            if p.metric_name == UNKNOWN and m_name != UNKNOWN:
                updates["metric_name"] = m_name
            if p.metric_value == UNKNOWN and m_val != UNKNOWN:
                updates["metric_value"] = m_val

        if p.model_name == UNKNOWN:
            for pat, name in _MODEL_PATTERNS:
                if pat.search(full_text):
                    updates["model_name"] = name
                    break

        if p.split_type == UNKNOWN:
            st = _extract_split_type(full_text)
            if st != UNKNOWN:
                updates["split_type"] = st

        if p.cross_validation == UNKNOWN:
            cv = _extract_cross_validation(full_text)
            if cv != UNKNOWN:
                updates["cross_validation"] = cv

        effective_modalities = p.modalities
        if not p.modalities:
            mods = [name for pat, name in _MODALITY_PATTERNS if pat.search(full_text)]
            if mods:
                updates["modalities"] = mods
                effective_modalities = mods

        if p.label_granularity == UNKNOWN:
            for pat, name in _LABEL_PATTERNS:
                if pat.search(full_text):
                    updates["label_granularity"] = name
                    break

        if not p.preprocessing_steps:
            prep = _extract_preprocessing(full_text, effective_modalities)
            if prep:
                updates["preprocessing_steps"] = prep

        if p.train_split == UNKNOWN or p.val_split == UNKNOWN or p.test_split == UNKNOWN:
            tr, va, te = _extract_splits(full_text)
            if p.train_split == UNKNOWN and tr != UNKNOWN:
                updates["train_split"] = tr
            if p.val_split == UNKNOWN and va != UNKNOWN:
                updates["val_split"] = va
            if p.test_split == UNKNOWN and te != UNKNOWN:
                updates["test_split"] = te

        enriched.append(p.model_copy(update=updates) if updates else p)
    return enriched


def score_paper(p: PaperRecord) -> float:
    """Completeness + recency score for top-N ranking. Higher = more informative.

    Papers with red flags are penalised so methodologically sound papers rank higher.
    A paper must have at least a metric and a model to be a useful SOTA candidate —
    papers missing both are heavily penalised so they never crowd out real results.
    """
    score = 0.0

    # Core SOTA signals — these are what make a paper useful as a benchmark reference
    has_metric_value = isinstance(p.metric_value, (int, float))
    has_metric_name = p.metric_name != UNKNOWN
    has_model = p.model_name != UNKNOWN

    if has_metric_value:
        score += 5.0   # Concrete number reported — most valuable
    if has_metric_name:
        score += 3.0   # At least tells us what was measured
    if has_model:
        score += 3.0   # Identifies what was compared

    # Heavy penalty for papers with neither metric nor model — these are
    # background/survey mentions, not SOTA results
    if not has_metric_name and not has_metric_value:
        score -= 8.0
    if not has_model:
        score -= 3.0

    if p.task != UNKNOWN:
        score += 2.0
    if p.split_type != UNKNOWN:
        score += 1.0
    if p.modalities:
        score += 0.5
    if p.label_granularity != UNKNOWN:
        score += 0.5
    if isinstance(p.year, int):
        score += max(0.0, min(1.0, (p.year - 2015) / 10.0))

    text = f"{p.title} {p.notes} {p.source_snippet}"
    if re.search(r"\b(review|survey|perspective|opinion)\b", p.title, re.I):
        score -= 4.0
    if re.search(r"\bfoundation\s+model\b", text, re.I) and re.search(
        r"\b(experiments?\s+(?:on|across)|outperform|benchmark|datasets?)\b",
        text,
        re.I,
    ):
        score += 3.0
    if re.search(r"\bfoundation\s+model\b", text, re.I) and re.search(
        r"\boutperform\w*.{0,80}\bbaselines?\b|\bbaselines?.{0,80}\boutperform\w*\b",
        text,
        re.I,
    ):
        score += 8.0
    if re.search(r"\bacross\s+\d+[^.]{0,80}\b(?:datasets?|tasks?)\b", text, re.I):
        score += 6.0

    evidence = dataset_use_evidence(p)
    if evidence in {"direct metadata mention", "linked from dataset page", "curated benchmark page"}:
        score += 10.0
    elif evidence == "cites dataset paper":
        score += 6.0
    elif "arXiv benchmark/model expansion" in p.notes:
        # Broad expansion is a recall aid for papers whose dataset list may be
        # buried in tables/appendices. Do not let it outrank candidates with
        # explicit target-dataset evidence when such candidates are available.
        score -= 8.0

    # Category penalty — RELATED WORK is always pushed below WEAK and STRONG
    _category_penalty = {
        "STRONG SOTA CANDIDATE": 0.0,
        "WEAK SOTA CANDIDATE": 3.0,
        "RELATED WORK ONLY": 12.0,
    }
    score -= _category_penalty.get(p.sota_category, 0.0)

    # Additional penalty per critical flag failed
    for flag in p.red_flags:
        tier = flag.split("|", 1)[0]
        score -= 2.0 if tier == "CRITICAL" else 0.5

    return score


def _mentions_target_dataset(p: PaperRecord) -> bool:
    text = f"{p.title} {p.notes} {p.source_snippet}"
    for dataset in p.datasets_used:
        if not dataset or dataset == UNKNOWN:
            continue
        compact = re.sub(r"[^A-Za-z0-9]", "", dataset)
        if compact.isupper() and len(compact) >= 2:
            if re.search(r"\b" + re.escape(dataset) + r"\b", text):
                return True
            continue
        needle = dataset.lower()
        haystack = text.lower()
        if needle in haystack:
            return True
        if needle.replace("-", " ").replace("_", " ") in haystack.replace("-", " ").replace("_", " "):
            return True
    return False


def dataset_use_evidence(p: PaperRecord) -> str:
    """Return the strongest evidence that this candidate uses the target dataset."""
    if _mentions_target_dataset(p):
        return "direct metadata mention"
    source_text = p.notes.lower()
    if "physionet dataset page" in source_text:
        return "linked from dataset page"
    if "paperswithcode" in source_text:
        return "curated benchmark page"
    if "citation chain" in source_text:
        return "cites dataset paper"
    return "no dataset-use evidence in metadata"


def top_n(papers: list[PaperRecord], n: int = 8) -> list[PaperRecord]:
    """Return the top-N SOTA candidates.

    For final red-flagged records, rank by target-dataset evidence, confidence,
    SOTA category, then extraction score. Before red flags are applied
    confidence/category are UNKNOWN, so this naturally falls back to score.
    """
    return sorted(papers, key=ranking_key, reverse=True)[:n]


def ranking_key(p: PaperRecord) -> tuple[float, float, int, float, float]:
    evidence_priority = {
        "direct metadata mention": 4,
        "linked from dataset page": 4,
        "curated benchmark page": 4,
        "cites dataset paper": 3,
        "no dataset-use evidence in metadata": 0,
    }.get(dataset_use_evidence(p), 0)

    category_priority = {
        "STRONG SOTA CANDIDATE": 3,
        "WEAK SOTA CANDIDATE": 2,
        "RELATED WORK ONLY": 1,
    }.get(p.sota_category, 0)

    confidence = p.confidence_score if isinstance(p.confidence_score, float) else -1.0
    return (
        float(evidence_priority),
        confidence,
        category_priority,
        score_paper(p),
        float(p.year) if isinstance(p.year, int) else 0.0,
    )
