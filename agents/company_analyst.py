from __future__ import annotations

from typing import Any

from memory.context import CompanyAnalystContext

from .model import ModelClient
from .schemas import CompanyAnalysis, Direction, jsonable


COMPANY_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thesis": {"type": "string"},
        "direction": {"type": "string", "enum": [item.value for item in Direction]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "horizon": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "invalidation_conditions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "thesis", "direction", "confidence", "horizon", "evidence_refs", "strengths",
        "weaknesses", "catalysts", "invalidation_conditions",
    ],
}


class CompanyAnalyst:
    """Builds a sourced company thesis from a permissioned context snapshot."""

    _required_fields = set(COMPANY_ANALYSIS_SCHEMA["required"])
    _list_fields = {
        "evidence_refs", "strengths", "weaknesses", "catalysts", "invalidation_conditions"
    }

    def __init__(self, model: ModelClient) -> None:
        self.model = model

    def analyze(self, context: CompanyAnalystContext) -> CompanyAnalysis:
        if not isinstance(context, CompanyAnalystContext):
            raise TypeError("CompanyAnalyst requires context from ContextGateway")
        if not context.records and not context.promoted_insights:
            raise ValueError("company analysis requires sourced records or promoted insights")

        allowed_refs = {record.ref for record in context.records} | {
            insight.ref for insight in context.promoted_insights
        }
        result = self.model.generate_json(
            instructions=(
                "You are the Company Analyst for a quantitative research system. Build a "
                "balanced, time-bounded company thesis using only the supplied regulatory "
                "records, company facts, and governance-promoted insights. Distinguish facts "
                "from interpretations. Never invent financial values, filings, guidance, "
                "comparables, prices, or events. Cite only supplied refs. If evidence is stale, "
                "incomplete, or contradictory, state that and reduce confidence. Do not propose "
                "position size or execute a trade."
            ),
            input_data={
                "symbol": context.symbol,
                "simulation_time": context.simulation_time.isoformat(),
                "company_records": [jsonable(record) for record in context.records],
                "promoted_insights": [jsonable(insight) for insight in context.promoted_insights],
            },
            schema=COMPANY_ANALYSIS_SCHEMA,
        )
        self._validate_output(result, allowed_refs)
        return CompanyAnalysis(
            symbol=context.symbol,
            thesis=result["thesis"].strip(),
            direction=Direction(result["direction"]),
            confidence=float(result["confidence"]),
            horizon=result["horizon"].strip(),
            evidence_refs=tuple(dict.fromkeys(result["evidence_refs"])),
            strengths=self._clean(result["strengths"]),
            weaknesses=self._clean(result["weaknesses"]),
            catalysts=self._clean(result["catalysts"]),
            invalidation_conditions=self._clean(result["invalidation_conditions"]),
        )

    def _validate_output(self, result: Any, allowed_refs: set[str]) -> None:
        if not isinstance(result, dict) or set(result) != self._required_fields:
            raise ValueError("company analyst output has missing or unexpected fields")
        for field in ("thesis", "horizon"):
            if not isinstance(result[field], str) or not result[field].strip():
                raise ValueError(f"{field} must be a non-empty string")
        if result["direction"] not in {item.value for item in Direction}:
            raise ValueError("direction is invalid")
        confidence = result["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
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
