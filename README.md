# SOTA Finder

SOTA Finder is a Streamlit tool for finding candidate state-of-the-art papers for biomedical machine-learning datasets. It retrieves papers from public scholarly sources, extracts structured fields from titles, abstracts, and available full text, and helps a human reviewer compare candidates.

The tool does not declare a winner automatically. Use it to gather candidates, then verify metrics, splits, labels, and preprocessing details in the original papers.

## Live App

Use SOTA Finder in your browser: https://sotafinder-l7nbysk9kjt8w3jgbpye9w.streamlit.app/

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd "SOTA Finder"
```

### 2. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Configure optional API keys

API keys are optional. Without them, SOTA Finder still runs with rule-based extraction.

Set one of these environment variables only if you want LLM-enhanced extraction:

macOS/Linux:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY = "your_gemini_api_key_here"
$env:ANTHROPIC_API_KEY = "your_anthropic_api_key_here"
```

Gemini is tried first when `GEMINI_API_KEY` is set. Anthropic is used when Gemini is not configured or Gemini fails.

Do not commit real API keys. `.env.example` contains placeholders only.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit.

### 6. Run the CLI

Run with built-in mock data:

```bash
python -m src.main
```

Run live retrieval for a dataset:

```bash
python -m src.main --dataset "MIT-BIH"
```

Reports are written under `outputs/`.

### 7. Run tests

```bash
pytest
```

## Troubleshooting

### Missing API keys

LLM extraction is optional. If no `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` is set, the app falls back to rule-based extraction and may show more fields as `UNKNOWN` or `not found`.

### Rate limits or empty search results

Live retrieval uses public services such as OpenAlex, Semantic Scholar, arXiv, Papers with Code, and publisher pages. If results are sparse or requests fail, wait a few minutes and retry. Free public APIs can rate-limit or temporarily return incomplete data.

### Inaccessible PDFs or full text

Some papers are paywalled, block automated downloads, or expose metadata without full text. In those cases, SOTA Finder can still use the title and abstract, but fields such as metric value, split type, preprocessing, and cross-validation may need manual verification.

### Windows script activation

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate `.venv` again.

## Development Notes

- `app.py` is the Streamlit entry point.
- `src/main.py` is the CLI entry point.
- `scripts/eval_gold_sota.py` runs live retrieval evaluation against gold cases and can take longer than the normal test suite.
- `outputs/`, virtual environments, caches, logs, and local `.env` files are ignored by git.
