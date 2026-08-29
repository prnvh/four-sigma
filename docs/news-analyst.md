# News Analyst

The News Analyst does not search the web or manufacture news. It accepts article
records collected by the future News Agent and returns an evidence-bound analysis.

## Accepted source record

Each article contains a stable `ref`, publisher `source`, original `url`, timezone-
aware `published_at`, original `title`, and provider-supplied `summary` or licensed
article text.

Suitable upstream sources include Alpha Vantage `NEWS_SENTIMENT`, licensed news
feeds, publisher APIs, or a team-maintained dataset. Provider sentiment is not
treated as truth; the analyst receives the underlying text and citation.

## Model and run command

The implementation uses the OpenAI Responses API with JSON Schema Structured
Outputs. Set `QFIRM_MODEL` explicitly to a compatible model available to your team.

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:QFIRM_MODEL = "your-model-id"
python run_news_analyst.py IBM articles.json
```

The JSON file must contain an array matching the accepted source record.

## Enforced properties

- Empty input is rejected.
- Confidence is restricted to `[0, 1]`.
- Every finding requires evidence.
- Any citation absent from the supplied articles is rejected.
- The module never executes trades.
