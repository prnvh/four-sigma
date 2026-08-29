from __future__ import annotations

from typing import Any

from .model import ModelClient
from .schemas import Direction, Evidence, Finding, jsonable


NEWS_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim": {"type": "string"},
        "direction": {"type": "string", "enum": [item.value for item in Direction]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "horizon": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claim", "direction", "confidence", "horizon", "evidence_refs", "risks"],
}


class NewsAnalyst:
    """Turns supplied, sourced articles into one evidence-bound market finding."""

    def __init__(self, model: ModelClient) -> None:
        self.model = model

    def analyze(self, symbol: str, articles: list[Evidence]) -> Finding:
        if not articles:
            raise ValueError("news analysis requires at least one sourced article")
        allowed_refs = {article.ref for article in articles}
        result = self.model.generate_json(
            instructions=(
                "You are the News Analyst for a quantitative research system. Analyze only "
                "the supplied articles. Assess materiality, likely market direction, time "
                "horizon, contradictions, and uncertainty. Never add a fact, event, price, "
                "or metric absent from the input. Cite only supplied evidence_refs. When the "
                "evidence is insufficient or conflicting, choose neutral and lower confidence."
            ),
            input_data={
                "symbol": symbol.strip().upper(),
                "articles": [jsonable(article) for article in articles],
            },
            schema=NEWS_FINDING_SCHEMA,
        )
        cited_refs = tuple(result["evidence_refs"])
        unknown_refs = set(cited_refs) - allowed_refs
        if unknown_refs:
            raise ValueError(f"news analyst cited unknown evidence: {sorted(unknown_refs)}")
        return Finding(
            agent="news_analyst",
            subject=symbol.strip().upper(),
            claim=result["claim"].strip(),
            direction=Direction(result["direction"]),
            confidence=float(result["confidence"]),
            horizon=result["horizon"].strip(),
            evidence_refs=cited_refs,
            risks=tuple(result["risks"]),
        )
