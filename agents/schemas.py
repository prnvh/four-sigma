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


class CompanyRecordType(str, Enum):
    REGULATORY_FILING = "regulatory_filing"
    FINANCIAL_FACT = "financial_fact"
    EARNINGS_RELEASE = "earnings_release"
    COMPANY_PROFILE = "company_profile"


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
class CompanyRecord:
    ref: str
    symbol: str
    source: str
    url: str
    knowledge_time: datetime
    record_type: CompanyRecordType
    label: str
    value: str
    period_end: str | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "ref": self.ref,
            "symbol": self.symbol,
            "source": self.source,
            "label": self.label,
            "value": self.value,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"company record {name} must be a non-empty string")
        if urlparse(self.url).scheme not in {"http", "https"}:
            raise ValueError("company record URL must use http or https")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("company record knowledge_time must be timezone-aware")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())


@dataclass(frozen=True, slots=True)
class PromotedInsight:
    ref: str
    symbol: str
    claim: str
    direction: Direction
    confidence: float
    evidence_refs: tuple[str, ...]
    knowledge_time: datetime
    valid_until: datetime | None = None

    def __post_init__(self) -> None:
        if not self.ref.strip() or not self.symbol.strip() or not self.claim.strip():
            raise ValueError("promoted insight identifiers and claim must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("promoted insight confidence must be between 0 and 1")
        if not self.evidence_refs:
            raise ValueError("promoted insight requires evidence references")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("promoted insight knowledge_time must be timezone-aware")
        if self.valid_until is not None:
            if self.valid_until.tzinfo is None:
                raise ValueError("promoted insight valid_until must be timezone-aware")
            if self.valid_until < self.knowledge_time:
                raise ValueError("valid_until cannot precede knowledge_time")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())


@dataclass(frozen=True, slots=True)
class CompanyAnalysis:
    symbol: str
    thesis: str
    direction: Direction
    confidence: float
    horizon: str
    evidence_refs: tuple[str, ...]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    catalysts: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.thesis.strip() or not self.horizon.strip():
            raise ValueError("company analysis symbol, thesis and horizon must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("company analysis confidence must be between 0 and 1")
        if not self.evidence_refs:
            raise ValueError("company analysis requires evidence references")


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
