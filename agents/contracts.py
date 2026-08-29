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
        frozen = _freeze(self.data)
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "data", frozen)

    @classmethod
    def parse(cls, value: object) -> Self:
        if not isinstance(value, Mapping):
            raise OutputValidationError("output must be an object")
        return cls(value)


class NewsAnalysisResult(StructuredAgentOutput):
    schema = NEWS_FINDING_SCHEMA
    contract_name = "finding"


class CompanyAnalysisResult(StructuredAgentOutput):
    schema = COMPANY_ANALYSIS_SCHEMA
    contract_name = "company_analysis"


class RiskAnalysisResult(StructuredAgentOutput):
    schema = RISK_ANALYSIS_SCHEMA
    contract_name = "risk_analysis"


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
