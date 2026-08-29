from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memory.capabilities import CAPABILITIES
from memory.context_gateway import CompanyAnalystContext

from .model import ModelClient
from .registry import COMPANY_ANALYST_V1, AgentSpec
from memory.types import CompanyAnalysis, jsonable


COMPANY_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company_thesis": {"type": "string"},
        "bull_case": {"type": "string"},
        "bear_case": {"type": "string"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "time_horizon": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "supports": {"type": "array", "items": {"type": "string"}},
        "contradicts": {"type": "array", "items": {"type": "string"}},
        "supersedes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "company_thesis", "bull_case", "bear_case", "catalysts", "risks",
        "confidence", "time_horizon", "evidence_refs", "supports",
        "contradicts", "supersedes",
    ],
}


class CompanyAnalyst:
    """Builds a sourced company thesis from a permissioned context snapshot."""

    _required_fields = set(COMPANY_ANALYSIS_SCHEMA["required"])
    _list_fields = {
        "catalysts", "risks", "evidence_refs", "supports", "contradicts",
        "supersedes",
    }

    def __init__(self, model: ModelClient, spec: AgentSpec = COMPANY_ANALYST_V1) -> None:
        if spec.name != "company_analyst":
            raise ValueError("CompanyAnalyst requires a company_analyst spec")
        self.model = model
        self.spec = spec

    def analyze(
        self,
        context: CompanyAnalystContext,
        *,
        requested_fields: Sequence[tuple[str, str]] = (),
    ) -> CompanyAnalysis:
        if not isinstance(context, CompanyAnalystContext):
            raise TypeError("CompanyAnalyst requires context from ContextGateway")
        CAPABILITIES.require_reads(
            self.spec.name,
            (
                ("company", "entity"),
                ("company", "records"),
                ("insights", "promoted"),
                ("events", "news"),
                ("market", "features"),
                *requested_fields,
            ),
        )
        if not (
            context.records
            or context.promoted_insights
            or context.recent_events
            or context.market_features
        ):
            raise ValueError("company analysis requires sourced evidence")

        insight_refs = {insight.ref for insight in context.promoted_insights}
        allowed_refs = (
            {record.ref for record in context.records}
            | insight_refs
            | {event.ref for event in context.recent_events}
            | {feature.ref for feature in context.market_features}
        )
        result = self.model.generate_json(
            instructions=self.spec.prompt,
            input_data={
                "symbol": context.symbol,
                "simulation_time": context.simulation_time.isoformat(),
                "company": jsonable(context.company),
                "company_facts": [jsonable(record) for record in context.records],
                "approved_insights": [
                    jsonable(insight) for insight in context.promoted_insights
                ],
                "recent_events": [jsonable(event) for event in context.recent_events],
                "historical_context": {
                    "company_records": [jsonable(record) for record in context.records],
                    "market_features": [
                        jsonable(feature) for feature in context.market_features
                    ],
                },
            },
            schema=COMPANY_ANALYSIS_SCHEMA,
        )
        self._validate_output(result, allowed_refs, insight_refs)
        return CompanyAnalysis(
            agent=self.spec.key,
            symbol=context.symbol,
            company_thesis=result["company_thesis"].strip(),
            bull_case=result["bull_case"].strip(),
            bear_case=result["bear_case"].strip(),
            catalysts=self._clean(result["catalysts"]),
            risks=self._clean(result["risks"]),
            confidence=float(result["confidence"]),
            time_horizon=result["time_horizon"].strip(),
            evidence_refs=tuple(dict.fromkeys(result["evidence_refs"])),
            supports=tuple(dict.fromkeys(result["supports"])),
            contradicts=tuple(dict.fromkeys(result["contradicts"])),
            supersedes=tuple(dict.fromkeys(result["supersedes"])),
        )

    def _validate_output(
        self, result: Any, allowed_refs: set[str], insight_refs: set[str]
    ) -> None:
        if not isinstance(result, dict) or set(result) != self._required_fields:
            raise ValueError("company analyst output has missing or unexpected fields")
        for field in ("company_thesis", "bull_case", "bear_case", "time_horizon"):
            if not isinstance(result[field], str) or not result[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        confidence = result["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("confidence must be numeric and between 0 and 1")
        for field in self._list_fields:
            if not isinstance(result[field], list) or not all(
                isinstance(item, str) for item in result[field]
            ):
                raise ValueError(f"{field} must be a list of strings")
        if not result["evidence_refs"]:
            raise ValueError("company analysis requires evidence references")
        unknown = set(result["evidence_refs"]) - allowed_refs
        if unknown:
            raise ValueError(f"company analyst cited unknown evidence: {sorted(unknown)}")
        relationship_refs = (
            set(result["supports"])
            | set(result["contradicts"])
            | set(result["supersedes"])
        )
        unknown_relationships = relationship_refs - insight_refs
        if unknown_relationships:
            raise ValueError(
                "company analyst related unknown insights: "
                f"{sorted(unknown_relationships)}"
            )
        if set(result["supports"]) & set(result["contradicts"]):
            raise ValueError("an insight cannot both support and contradict the same insight")

    @staticmethod
    def _clean(values: list[str]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())
