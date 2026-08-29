# Company Analyst

The Company Analyst builds a balanced, time-bounded thesis from normalized company
records and governance-promoted insights. It does not fetch data, size positions,
read portfolios, or execute trades.

## Permitted context

`ContextGateway` exposes only records matching the requested symbol that were
knowable at the simulation time, plus non-expired promoted insights. Raw news,
private working memory and portfolio state are not included.

Supported company record types are regulatory filings, financial facts, earnings
releases and company profiles. Each record preserves its provider, original URL,
knowledge time, label, value and reporting period.

## Recommended authoritative upstream sources

- US issuers: SEC EDGAR submissions and XBRL Company Facts APIs.
- Indian issuers: NSE/BSE corporate filing feeds and issuer-filed disclosures.
- Issuer investor-relations releases when linked to their original publication.

This commit implements the analyst and source contract, not the data pullers.

## Output

The strict structured result includes thesis, direction, confidence, horizon,
citations, strengths, weaknesses, catalysts and thesis-invalidation conditions.
Unknown citations and malformed output are rejected.

## Run

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "your-model-id"
python run_company_analyst.py IBM company-context.json --as-of 2026-08-29T12:00:00+00:00
```

The input JSON object contains `company_records` and `promoted_insights` arrays.
