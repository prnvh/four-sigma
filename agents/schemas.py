from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class Evidence:
    ref: str
    source: str
    url: str
    published_at: datetime
    title: str
    summary: str = ""


@dataclass(frozen=True, slots=True)
class Finding:
    agent: str
    subject: str
    claim: str
    direction: Direction
    confidence: float
    horizon: str
    evidence_refs: tuple[str, ...]
    risks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not self.evidence_refs:
            raise ValueError("a finding requires at least one evidence reference")


def jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
