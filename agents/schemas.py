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


class MarketFeatureType(str, Enum):
    RETURN_5D = "return_5d"
    RETURN_20D = "return_20d"
    RETURN_60D = "return_60d"
    VOLUME_RATIO_20D = "volume_ratio_20d"
    VOLATILITY_20D = "volatility_20d"
    RELATIVE_STRENGTH_20D = "relative_strength_20d"


class RiskCategory(str, Enum):
    BUSINESS = "business"
    FINANCIAL = "financial"
    LIQUIDITY = "liquidity"
    MARKET = "market"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    GOVERNANCE = "governance"
    EVENT = "event"
    SENTIMENT = "sentiment"
    DATA_MODEL = "data_model"


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
class MarketFeature:
    ref: str
    symbol: str
    source: str
    url: str
    knowledge_time: datetime
    feature: MarketFeatureType
    value: float
    unit: str

    def __post_init__(self) -> None:
        for name, value in {
            "ref": self.ref, "symbol": self.symbol, "source": self.source, "unit": self.unit
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"market feature {name} must be a non-empty string")
        if urlparse(self.url).scheme not in {"http", "https"}:
            raise ValueError("market feature URL must use http or https")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("market feature knowledge_time must be timezone-aware")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("market feature value must be numeric")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())


@dataclass(frozen=True, slots=True)
class CompanyAnalysis:
    symbol: str
    thesis: str
    fundamental_direction: Direction
    fundamental_confidence: float
    momentum_direction: Direction
    momentum_score: float
    momentum_confidence: float
    momentum_horizon: str
    evidence_refs: tuple[str, ...]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    catalysts: tuple[str, ...]
    momentum_drivers: tuple[str, ...]
    momentum_risks: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.thesis.strip() or not self.momentum_horizon.strip():
            raise ValueError("company analysis identifiers and horizon must be non-empty")
        if not 0 <= self.fundamental_confidence <= 1 or not 0 <= self.momentum_confidence <= 1:
            raise ValueError("company analysis confidence must be between 0 and 1")
        if not -1 <= self.momentum_score <= 1:
            raise ValueError("momentum_score must be between -1 and 1")
        if not self.evidence_refs:
            raise ValueError("company analysis requires evidence references")


@dataclass(frozen=True, slots=True)
class CompanyAnalysisRecord:
    ref: str
    analysis: CompanyAnalysis
    knowledge_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError("company analysis record ref must be non-empty")
        if not isinstance(self.analysis, CompanyAnalysis):
            raise TypeError("analysis must be CompanyAnalysis")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("company analysis knowledge_time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class OutcomeDefinition:
    horizon_days: int
    success_return_pct: float
    failure_return_pct: float

    def __post_init__(self) -> None:
        if isinstance(self.horizon_days, bool) or not isinstance(self.horizon_days, int):
            raise TypeError("horizon_days must be an integer")
        if not 1 <= self.horizon_days <= 365:
            raise ValueError("horizon_days must be between 1 and 365")
        if self.failure_return_pct >= self.success_return_pct:
            raise ValueError("failure threshold must be below success threshold")


@dataclass(frozen=True, slots=True)
class RiskFactor:
    category: RiskCategory
    probability_pct: float
    severity: int
    impact: str
    evidence_refs: tuple[str, ...]
    mitigants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RiskAnalysis:
    symbol: str
    horizon_days: int
    overall_risk_score: float
    success_probability_pct: float
    neutral_probability_pct: float
    failure_probability_pct: float
    risk_factors: tuple[RiskFactor, ...]
    hidden_assumptions: tuple[str, ...]
    second_order_effects: tuple[str, ...]
    success_conditions: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    coverage_gaps: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        probabilities = (
            self.success_probability_pct,
            self.neutral_probability_pct,
            self.failure_probability_pct,
        )
        if any(not 0 <= value <= 100 for value in probabilities):
            raise ValueError("outcome probabilities must be between 0 and 100")
        if abs(sum(probabilities) - 100) > 0.01:
            raise ValueError("outcome probabilities must sum to 100")
        if not 0 <= self.overall_risk_score <= 100:
            raise ValueError("overall_risk_score must be between 0 and 100")
        if not self.evidence_refs:
            raise ValueError("risk analysis requires evidence references")


@dataclass(frozen=True, slots=True)
class MarketState:
    ref: str
    symbol: str
    source: str
    url: str
    event_time: datetime
    knowledge_time: datetime
    simulation_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int

    def __post_init__(self) -> None:
        for name, value in {
            "ref": self.ref,
            "symbol": self.symbol,
            "source": self.source,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"market state {name} must be a non-empty string")
        if urlparse(self.url).scheme not in {"http", "https"}:
            raise ValueError("market state URL must use http or https")
        for name, value in {
            "event_time": self.event_time,
            "knowledge_time": self.knowledge_time,
            "simulation_time": self.simulation_time,
        }.items():
            if value.tzinfo is None:
                raise ValueError(f"market state {name} must be timezone-aware")
        if self.knowledge_time > self.simulation_time:
            raise ValueError("market state knowledge_time cannot be after simulation_time")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())


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
