from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import unquote

from memory.capabilities import CAPABILITIES
from memory.context_gateway import NewsAnalystContext

from .model import ModelClient
from .registry import NEWS_ANALYST_V1, AgentSpec
from memory.types import Direction, Finding, jsonable


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

    def __init__(self, model: ModelClient, spec: AgentSpec = NEWS_ANALYST_V1) -> None:
        if spec.name != "news_analyst":
            raise ValueError("NewsAnalyst requires a news_analyst spec")
        self.model = model
        self.spec = spec

    def analyze(
        self,
        context: NewsAnalystContext,
        *,
        requested_fields: Sequence[tuple[str, str]] = (),
    ) -> Finding:
        if not isinstance(context, NewsAnalystContext):
            raise TypeError("NewsAnalyst requires context from ContextGateway")
        CAPABILITIES.require_reads(self.spec.name, (("events", "news"), *requested_fields))
        if not context.articles:
            raise ValueError("news analysis requires at least one sourced article")
        allowed_refs = {article.ref for article in context.articles}
        result = self.model.generate_json(
            instructions=self.spec.prompt,
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
        cited_refs = _bound_evidence_refs(result["evidence_refs"], context)
        unknown_refs = set(cited_refs) - allowed_refs
        if not cited_refs or unknown_refs:
            raise ValueError(f"news analyst cited unknown evidence: {sorted(unknown_refs)}")
        return Finding(
            agent=self.spec.key,
            subject=context.symbol,
            claim=result["claim"].strip(),
            direction=Direction(result["direction"]),
            confidence=float(result["confidence"]),
            horizon=result["horizon"].strip(),
            evidence_refs=cited_refs,
            risks=tuple(str(risk).strip() for risk in result["risks"] if str(risk).strip()),
        )


def _bound_evidence_refs(cited: Sequence[str], context: NewsAnalystContext) -> tuple[str, ...]:
    allowed = {article.ref: article for article in context.articles}
    resolved: list[str] = []
    for raw in cited:
        match = _bound_ref(raw, allowed)
        if match is not None and match not in resolved:
            resolved.append(match)
    return tuple(resolved)


def _bound_ref(cited: str, allowed: dict[str, object]) -> str | None:
    if cited in allowed:
        return cited
    cited_norm = unquote(cited)
    cited_url = cited_norm.removeprefix("gdelt:")
    for ref, article in allowed.items():
        url = getattr(article, "url", "")
        if cited_norm == unquote(ref) or cited_url == url or cited_url == unquote(ref).removeprefix("gdelt:"):
            return ref
        if url and (cited_url.endswith(url) or url.endswith(cited_url)):
            return ref
    return None
