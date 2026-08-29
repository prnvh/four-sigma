from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Self

from memory.types import TradeSide

from .company_analyst import COMPANY_ANALYSIS_SCHEMA
from .news_analyst import NEWS_FINDING_SCHEMA
from .risk_analyst import RISK_ANALYSIS_SCHEMA


class OutputValidationError(ValueError):
    """A model response did not match its declared output contract."""


def _validate(value: object, schema: Mapping[str, Any], path: str = "output") -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, Mapping):
            raise OutputValidationError(f"{path} must be an object")
        properties = schema.get("properties", {})
        missing = set(schema.get("required", ())) - set(value)
        unknown = set(value) - set(properties)
        if missing:
            raise OutputValidationError(f"{path} is missing fields: {sorted(missing)}")
        if schema.get("additionalProperties") is False and unknown:
            raise OutputValidationError(f"{path} has unexpected fields: {sorted(unknown)}")
        for key, item in value.items():
            child = properties.get(key)
            if child is not None:
                _validate(item, child, f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list):
            raise OutputValidationError(f"{path} must be an array")
        if len(value) < schema.get("minItems", 0):
            raise OutputValidationError(f"{path} has too few items")
        child = schema.get("items", {})
        for index, item in enumerate(value):
            _validate(item, child, f"{path}[{index}]")
    elif kind == "string":
        if not isinstance(value, str):
            raise OutputValidationError(f"{path} must be a string")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OutputValidationError(f"{path} must be a number")
        if not math.isfinite(value):
            raise OutputValidationError(f"{path} must be finite")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutputValidationError(f"{path} must be an integer")

    if "enum" in schema and value not in schema["enum"]:
        raise OutputValidationError(f"{path} is not an allowed value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise OutputValidationError(f"{path} is below its minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise OutputValidationError(f"{path} is above its maximum")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class StructuredAgentOutput:
    """Validated, immutable model output safe for downstream consumers."""

    data: Mapping[str, object]
    schema: ClassVar[Mapping[str, Any]]
    contract_name: ClassVar[str]

    def __post_init__(self) -> None:
        _validate(self.data, self.schema)
        self.validate_semantics(self.data)
        frozen = _freeze(self.data)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "data", frozen)

    @classmethod
    def parse(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise OutputValidationError("output must be an object")
        return cls(value)

    @classmethod
    def validate_semantics(cls, data: Mapping[str, object]) -> None:
        """Validate invariants JSON Schema cannot express by itself."""
        _reject_blank_strings(data)


class NewsAnalysisResult(StructuredAgentOutput):
    schema = NEWS_FINDING_SCHEMA
    contract_name = "finding"

    @classmethod
    def validate_semantics(cls, data: Mapping[str, object]) -> None:
        super().validate_semantics(data)
        _require_unique(data["evidence_refs"], "output.evidence_refs")


class CompanyAnalysisResult(StructuredAgentOutput):
    schema = COMPANY_ANALYSIS_SCHEMA
    contract_name = "company_analysis"

    @classmethod
    def validate_semantics(cls, data: Mapping[str, object]) -> None:
        super().validate_semantics(data)
        for field in ("evidence_refs", "supports", "contradicts", "supersedes"):
            _require_unique(data[field], f"output.{field}")
        if set(data["supports"]) & set(data["contradicts"]):
            raise OutputValidationError(
                "output cannot support and contradict the same insight"
            )


class RiskAnalysisResult(StructuredAgentOutput):
    schema = RISK_ANALYSIS_SCHEMA
    contract_name = "risk_analysis"

    @classmethod
    def validate_semantics(cls, data: Mapping[str, object]) -> None:
        super().validate_semantics(data)
        probabilities = (
            data["success_probability_pct"],
            data["neutral_probability_pct"],
            data["failure_probability_pct"],
        )
        if abs(sum(probabilities) - 100) > 0.01:
            raise OutputValidationError("outcome probabilities must total 100")
        factors = data["risk_factors"]
        assessed = [factor["category"] for factor in factors]
        gaps = data["coverage_gaps"]
        _require_unique(assessed, "output.risk_factors categories")
        _require_unique(gaps, "output.coverage_gaps")
        if set(assessed) & set(gaps):
            raise OutputValidationError("a risk category cannot be assessed and a gap")
        expected = set(RISK_ANALYSIS_SCHEMA["properties"]["coverage_gaps"]["items"]["enum"])
        if set(assessed) | set(gaps) != expected:
            raise OutputValidationError(
                "every risk category must be assessed or declared a coverage gap"
            )
        _require_unique(data["evidence_refs"], "output.evidence_refs")
        for index, factor in enumerate(factors):
            _require_unique(
                factor["evidence_refs"], f"output.risk_factors[{index}].evidence_refs"
            )


TRADE_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "instrument": {"type": "string"},
        "direction": {"type": "string", "enum": [side.value for side in TradeSide]},
        "thesis_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "horizon": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "entry_conditions": {"type": "array", "items": {"type": "string"}},
        "exit_conditions": {"type": "array", "items": {"type": "string"}},
        "proposed_size": {"type": "number", "minimum": 0},
    },
    "required": [
        "instrument", "direction", "thesis_refs", "horizon", "confidence",
        "entry_conditions", "exit_conditions", "proposed_size",
    ],
}


class TradeProposalResult(StructuredAgentOutput):
    schema = TRADE_PROPOSAL_SCHEMA
    contract_name = "trade_proposal"

    @classmethod
    def validate_semantics(cls, data: Mapping[str, object]) -> None:
        super().validate_semantics(data)
        _require_unique(data["thesis_refs"], "output.thesis_refs")
        direction = data["direction"]
        size = data["proposed_size"]
        if direction == TradeSide.NO_TRADE.value and size != 0:
            raise OutputValidationError("no_trade output must have proposed_size 0")
        if direction != TradeSide.NO_TRADE.value and size == 0:
            raise OutputValidationError("long and short outputs require positive size")


def _reject_blank_strings(value: object, path: str = "output") -> None:
    if isinstance(value, str):
        if not value.strip():
            raise OutputValidationError(f"{path} must not be blank")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_blank_strings(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_blank_strings(item, f"{path}[{index}]")


def _require_unique(value: object, path: str) -> None:
    assert isinstance(value, list)
    if len(value) != len(set(value)):
        raise OutputValidationError(f"{path} must not contain duplicates")
