# AI Risk Analyst

The AI Risk Analyst is a qualitative challenge layer for external listed-company
research. It reads only point-in-time company analysis, company records, approved
insights and deterministic market features supplied through the context gateway.
It searches for business, financial, liquidity, market, regulatory, operational,
governance, event, sentiment and data/model risks without inventing missing facts.
It estimates success, neutral and failure percentages against explicit return
thresholds and a fixed horizon supplied by the caller. The three estimates must sum
to 100%, and unsupported risk categories must be declared as coverage gaps. It also
returns hidden assumptions, second-order effects, mitigants and conditions that
would support or invalidate the thesis. The output is advisory only: it cannot read
portfolio state, approve a trade, bypass the deterministic risk engine or guarantee
an outcome. Its greatest potential is ranking and comparing risk consistently over a
large company universe, with later backtests used to measure probability calibration.

## Sources and methodology

- Company-specific risk evidence should originate from regulatory filings, especially
  risk factors, MD&A, market-risk disclosures, financial statements and audit notes.
- Market inputs are deterministic, point-in-time features produced upstream.
- Model output uses a strict JSON Schema and every identified risk needs a supplied
  evidence reference.

Model-estimated probabilities are hypotheses. They are not historical frequencies
until calibration tests demonstrate that predicted buckets match observed outcomes.

## Run

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "your-model-id"
python run_risk_analyst.py IBM risk-context.json `
  --horizon-days 30 --success-return-pct 10 --failure-return-pct -10
```
