from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse


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
    symbols: tuple[str, ...] = ()
    knowledge_time: datetime | None = None

    def __post_init__(self) -> None:
        required = {"ref": self.ref, "source": self.source, "title": self.title}
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence {name} must be a non-empty string")
        if urlparse(self.url).scheme not in {"http", "https"}:
            raise ValueError("evidence URL must use http or https")
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        known_at = self.knowledge_time or self.published_at
        if known_at.tzinfo is None:
            raise ValueError("knowledge_time must be timezone-aware")
        object.__setattr__(self, "knowledge_time", known_at)
        object.__setattr__(self, "symbols", tuple(symbol.strip().upper() for symbol in self.symbols))


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
        for name, value in {"claim": self.claim, "horizon": self.horizon}.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"finding {name} must be a non-empty string")


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
