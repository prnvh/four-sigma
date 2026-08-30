# QFIRM

A governed multi-agent quantitative research project, implemented one reviewable
module per commit.

## Implemented modules

The first module reads sourced article records through a permissioned, point-in-time
context gateway and produces an evidence-bound structured finding. It rejects empty
input, future or unrelated records, and citations that were not supplied.

See `docs/news-analyst.md` for its input contract and source policy.

The Company Analyst evaluates external listed companies using point-in-time company
records, promoted insights and deterministic market features. It returns separate
fundamental and probabilistic stock-momentum outlooks. See `docs/company-analyst.md`.

The AI Risk Analyst challenges company theses, covers ten risk domains, and estimates
explicit success/neutral/failure scenarios without trade authority. See
`docs/risk-analyst.md`.

## Run it

Requires Python 3.11+ and an OpenAI API key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "a-model-available-in-your-account"
python -m agents.run_news_analyst IBM articles.json
```

The command prints structured JSON and does not execute trades.

The end-to-end backtest parallelizes independent symbol tracks and caches identical
model requests for reproducible, fast reruns:

```powershell
python -m agents.run_backtest --start 2026-01-01 --end 2026-03-31 `
  --universe AAPL MSFT NVDA AMZN --interval 1h --workers 8
```

Cached responses live under `.qfirm-cache/`. Use `--no-model-cache` only when a
fresh model sample is intentionally part of the experiment.

## Test it without API keys

```powershell
python -m unittest discover -s tests -v
```

Tests use clearly labelled synthetic fixtures and never call external services.

## Paper-run dashboard

The dashboard automatically follows the newest log in `.qfirm-cache/runs/` and
refreshes every ten seconds while the run is active:

```powershell
npm run dev
```

Open `http://127.0.0.1:8000`. To follow a specific log, set
`QFIRM_RUN_LOG` before starting the server.

For a lightweight Vercel upload, export the current read-only snapshot and deploy:

```powershell
npm run snapshot
npm run deploy
```

The deployment includes the dashboard snapshot, not `.env`, model caches, or API
keys. Run `npm run snapshot` again before a later upload to publish fresher results.
