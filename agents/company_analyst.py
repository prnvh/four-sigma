from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memory.capabilities import CAPABILITIES
from memory.context_gateway import CompanyAnalystContext

from .model import ModelClient
from .registry import COMPANY_ANALYST_V1, AgentSpec
from memory.types import CompanyAnalysis, Direction, jsonable


COMPANY_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thesis": {"type": "string"},
        "fundamental_direction": {"type": "string", "enum": [item.value for item in Direction]},
        "fundamental_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "momentum_direction": {"type": "string", "enum": [item.value for item in Direction]},
        "momentum_score": {"type": "number", "minimum": -1, "maximum": 1},
        "momentum_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "momentum_horizon": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "momentum_drivers": {"type": "array", "items": {"type": "string"}},
        "momentum_risks": {"type": "array", "items": {"type": "string"}},
        "invalidation_conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "thesis", "fundamental_direction", "fundamental_confidence", "momentum_direction",
        "momentum_score", "momentum_confidence", "momentum_horizon", "evidence_refs",
        "strengths", "weaknesses", "catalysts", "momentum_drivers", "momentum_risks",
        "invalidation_conditions",
    ],
}


class CompanyAnalyst:
    """Builds a sourced company thesis from a permissioned context snapshot."""

    _required_fields = set(COMPANY_ANALYSIS_SCHEMA["required"])
    _list_fields = {
        "evidence_refs", "strengths", "weaknesses", "catalysts", "momentum_drivers",
        "momentum_risks", "invalidation_conditions"
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
            (("company", "records"), ("insights", "promoted"), ("market", "features"), *requested_fields),
        )
        if not context.records and not context.promoted_insights and not context.market_features:
            raise ValueError("company analysis requires sourced company, insight, or market evidence")

        allowed_refs = {record.ref for record in context.records} | {
            insight.ref for insight in context.promoted_insights
        } | {feature.ref for feature in context.market_features}
        result = self.model.generate_json(
            instructions=self.spec.prompt,
            input_data={
                "symbol": context.symbol,
                "simulation_time": context.simulation_time.isoformat(),
                "company_records": [jsonable(record) for record in context.records],
                "promoted_insights": [jsonable(insight) for insight in context.promoted_insights],
                "market_features": [jsonable(feature) for feature in context.market_features],
            },
            schema=COMPANY_ANALYSIS_SCHEMA,
        )
        self._validate_output(result, allowed_refs)
        return CompanyAnalysis(
            agent=self.spec.key,
            symbol=context.symbol,
            thesis=result["thesis"].strip(),
            fundamental_direction=Direction(result["fundamental_direction"]),
            fundamental_confidence=float(result["fundamental_confidence"]),
            momentum_direction=Direction(result["momentum_direction"]),
            momentum_score=float(result["momentum_score"]),
            momentum_confidence=float(result["momentum_confidence"]),
            momentum_horizon=result["momentum_horizon"].strip(),
            evidence_refs=tuple(dict.fromkeys(result["evidence_refs"])),
            strengths=self._clean(result["strengths"]),
            weaknesses=self._clean(result["weaknesses"]),
            catalysts=self._clean(result["catalysts"]),
            momentum_drivers=self._clean(result["momentum_drivers"]),
            momentum_risks=self._clean(result["momentum_risks"]),
            invalidation_conditions=self._clean(result["invalidation_conditions"]),
        )

    def _validate_output(self, result: Any, allowed_refs: set[str]) -> None:
        if not isinstance(result, dict) or set(result) != self._required_fields:
            raise ValueError("company analyst output has missing or unexpected fields")
        for field in ("thesis", "momentum_horizon"):
            if not isinstance(result[field], str) or not result[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        for field in ("fundamental_direction", "momentum_direction"):
            if result[field] not in {item.value for item in Direction}:
                raise ValueError(f"{field} is invalid")
        for field in ("fundamental_confidence", "momentum_confidence"):
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be numeric and between 0 and 1")
        score = result["momentum_score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not -1 <= score <= 1:
            raise ValueError("momentum_score must be numeric and between -1 and 1")
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

    @staticmethod
    def _clean(values: list[str]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())
