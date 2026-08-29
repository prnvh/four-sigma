# QFIRM

A governed multi-agent quantitative research project, implemented one reviewable
module per commit.

## Implemented modules

The first module reads sourced article records through a permissioned, point-in-time
context gateway and produces an evidence-bound structured finding. It rejects empty
input, future or unrelated records, and citations that were not supplied.

See `docs/news-analyst.md` for its input contract and source policy.

The Company Analyst reads only point-in-time company records and promoted insights
through its own least-privilege context. See `docs/company-analyst.md`.

## Run it

Requires Python 3.11+ and an OpenAI API key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "a-model-available-in-your-account"
python run_news_analyst.py IBM articles.json
```

The command prints structured JSON and does not execute trades.

## Test it without API keys

```powershell
python -m unittest discover -s tests -v
```

Tests use clearly labelled synthetic fixtures and never call external services.

## Architecture layout

Open `web/index.html` to view the responsive architecture reference.
