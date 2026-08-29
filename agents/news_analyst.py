from __future__ import annotations

from typing import Any

from memory.context import NewsAnalystContext

from .model import ModelClient
from .schemas import Direction, Finding, jsonable


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

    def analyze(self, context: NewsAnalystContext) -> Finding:
        if not isinstance(context, NewsAnalystContext):
            raise TypeError("NewsAnalyst requires context from ContextGateway")
        if not context.articles:
            raise ValueError("news analysis requires at least one sourced article")
        allowed_refs = {article.ref for article in context.articles}
        result = self.model.generate_json(
            instructions=(
                "You are the News Analyst for a quantitative research system. Analyze only "
                "the supplied articles. Assess materiality, likely market direction, time "
                "horizon, contradictions, and uncertainty. Never add a fact, event, price, "
                "or metric absent from the input. Cite only supplied evidence_refs. When the "
                "evidence is insufficient or conflicting, choose neutral and lower confidence."
            ),
            input_data={
                "symbol": context.symbol,
                "simulation_time": context.simulation_time.isoformat(),
                "articles": [jsonable(article) for article in context.articles],
            },
            schema=NEWS_FINDING_SCHEMA,
        )
        if not isinstance(result, dict):
            raise ValueError("news analyst model output must be an object")
        required = {"claim", "direction", "confidence", "horizon", "evidence_refs", "risks"}
        if set(result) != required:
            raise ValueError("news analyst model output has missing or unexpected fields")
        if not isinstance(result["claim"], str) or not result["claim"].strip():
            raise ValueError("claim must be a non-empty string")
        if not isinstance(result["horizon"], str) or not result["horizon"].strip():
            raise ValueError("horizon must be a non-empty string")
        if not isinstance(result["confidence"], (int, float)) or isinstance(
            result["confidence"], bool
        ):
            raise ValueError("confidence must be numeric")
        if result["direction"] not in {item.value for item in Direction}:
            raise ValueError("direction is invalid")
        if not isinstance(result["evidence_refs"], list) or not all(
            isinstance(ref, str) for ref in result["evidence_refs"]
        ):
            raise ValueError("evidence_refs must be a list of strings")
        if not isinstance(result["risks"], list) or not all(
            isinstance(risk, str) for risk in result["risks"]
        ):
            raise ValueError("risks must be a list of strings")
        cited_refs = tuple(dict.fromkeys(result["evidence_refs"]))
        unknown_refs = set(cited_refs) - allowed_refs
        if unknown_refs:
            raise ValueError(f"news analyst cited unknown evidence: {sorted(unknown_refs)}")
        return Finding(
            agent="news_analyst",
            subject=context.symbol,
            claim=result["claim"].strip(),
            direction=Direction(result["direction"]),
            confidence=float(result["confidence"]),
            horizon=result["horizon"].strip(),
            evidence_refs=cited_refs,
            risks=tuple(str(risk).strip() for risk in result["risks"] if str(risk).strip()),
        )
