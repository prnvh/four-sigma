# Company Analyst

The Company Analyst evaluates an external listed company and produces both a
fundamental thesis and a probabilistic stock-momentum outlook. It uses normalized
company records, deterministic market features and governance-promoted insights.
It does not fetch data, size positions, read portfolios, or execute trades.

## Permitted context

`ContextGateway` exposes only records matching the requested symbol that were
knowable at the simulation time, plus non-expired promoted insights and point-in-time
market features. Raw news, private working memory and portfolio state are excluded.

Supported company record types are regulatory filings, financial facts, earnings
releases and company profiles. Each record preserves its provider, original URL,
knowledge time, label, value and reporting period.

## Recommended authoritative upstream sources

- US issuers: SEC EDGAR submissions and XBRL Company Facts APIs.
- Indian issuers: NSE/BSE corporate filing feeds and issuer-filed disclosures.
- Issuer investor-relations releases when linked to their original publication.

This commit implements the analyst and source contract, not the data pullers.

## Output

The strict result separates fundamental direction from stock momentum. Momentum
contains direction, a score from -1 to +1, confidence, horizon, evidence-backed
drivers and risks. It is an uncertain outlook, not a guaranteed prediction or price
target. The remainder includes thesis, strengths, weaknesses, catalysts and
invalidation conditions. Unknown citations and malformed output are rejected.

## Run

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "your-model-id"
python run_company_analyst.py IBM company-context.json --as-of 2026-08-29T12:00:00+00:00
```

The input JSON object contains `company_records`, `promoted_insights`, and
`market_features` arrays. Market features are calculated upstream and may include
5/20/60-day return, 20-day volume ratio, volatility and relative strength.
