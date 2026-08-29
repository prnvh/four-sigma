from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any
from types import MappingProxyType
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class CanonicalId:
    value: str

    def __post_init__(self) -> None:
        if type(self) is CanonicalId:
            raise TypeError("use a concrete ID type")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError(f"{type(self).__name__} must be a non-empty string")

    def __str__(self) -> str:
        return self.value


class AgentId(CanonicalId):
    __slots__ = ()


class RunId(CanonicalId):
    __slots__ = ()


class EventId(CanonicalId):
    __slots__ = ()


class AuditEventId(CanonicalId):
    __slots__ = ()


class InsightId(CanonicalId):
    __slots__ = ()


class ProposalId(CanonicalId):
    __slots__ = ()


class TradeCandidateId(CanonicalId):
    __slots__ = ()


class FillId(CanonicalId):
    __slots__ = ()


class EvidenceId(CanonicalId):
    __slots__ = ()


class DecisionId(CanonicalId):
    __slots__ = ()


class EntityId(CanonicalId):
    __slots__ = ()


class ContextSnapshotId(CanonicalId):
    __slots__ = ()


class MissingKnowledgeTime(ValueError):
    """Historical data was constructed without a KnowledgeTime."""


@dataclass(frozen=True, slots=True)
class Instant:
    value: datetime

    def __post_init__(self) -> None:
        if type(self) is Instant:
            raise TypeError(
                "use EventTime, KnowledgeTime, SimulationTime, or CreatedAt"
            )
        if not isinstance(self.value, datetime):
            raise TypeError(
                f"{type(self).__name__} requires datetime, got {type(self.value).__name__}"
            )
        if self.value.tzinfo is None:
            raise ValueError(f"{type(self).__name__} must be timezone-aware")
        utc = self.value.astimezone(timezone.utc)
        object.__setattr__(self, "value", utc)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.value <= other.value

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Instant):
            return NotImplemented
        return self.value < other.value


class EventTime(Instant):
    """When the fact occurred in the world.

    Earnings quarter-end, trade print, news occurrence.
    Never use this as the as-of filter.
    """

    __slots__ = ()


class KnowledgeTime(Instant):
    """When the fact became knowable.

    Earnings release, publication, revision availability.
    Historical queries and agent context filter on this.
    """

    __slots__ = ()


class SimulationTime(Instant):
    """Current time of the deterministic simulation clock.

    Agents may only observe records with knowledge_time <= simulation_time.
    """

    __slots__ = ()


class CreatedAt(Instant):
    """When the record was written into this system.

    Distinct from event_time and knowledge_time. Must be supplied
    explicitly — never read from the wall clock.
    """

    __slots__ = ()


def parse_knowledge_time(value: object) -> KnowledgeTime:
    if value is None:
        raise MissingKnowledgeTime("historical data requires knowledge_time")
    if isinstance(value, KnowledgeTime):
        return value
    if isinstance(value, Instant):
        raise MissingKnowledgeTime(
            f"historical data requires KnowledgeTime, got {type(value).__name__}"
        )
    if isinstance(value, datetime):
        return KnowledgeTime(value)
    raise TypeError(
        f"knowledge_time must be KnowledgeTime or datetime, got {type(value).__name__}"
    )


def visible_as_of(
    knowledge_time: KnowledgeTime, simulation_time: SimulationTime
) -> bool:
    """True when the record was knowable at simulation_time."""
    return knowledge_time <= simulation_time


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoricalRecord:
    """Any record of the past. knowledge_time has no default."""

    knowledge_time: KnowledgeTime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "knowledge_time", parse_knowledge_time(self.knowledge_time)
        )

    def visible_as_of(self, simulation_time: SimulationTime) -> bool:
        return visible_as_of(self.knowledge_time, simulation_time)


class InsightStatus(str, Enum):
    ACTIVE = "active"
    RETRACTED = "retracted"


@dataclass(frozen=True, slots=True, kw_only=True)
class InsightRevision:
    """The versioning instructions carried by an approved proposal."""

    insight_id: InsightId
    value: object
    status: InsightStatus = InsightStatus.ACTIVE
    valid_until: SimulationTime | None = None
    supports: tuple[InsightId, ...] = ()
    contradicts: tuple[InsightId, ...] = ()
    supersedes: tuple[InsightId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.insight_id, InsightId):
            raise TypeError("insight_id must be InsightId")
        if not isinstance(self.status, InsightStatus):
            raise TypeError("status must be InsightStatus")
        if self.valid_until is not None and not isinstance(
            self.valid_until, SimulationTime
        ):
            raise TypeError("valid_until must be SimulationTime or None")
        _validate_insight_relations(
            self.insight_id, self.supports, self.contradicts, self.supersedes
        )
        object.__setattr__(self, "value", deepcopy(self.value))


@dataclass(frozen=True, slots=True, kw_only=True)
class InsightVersion:
    """One immutable version of a logical shared insight."""

    insight_id: InsightId
    entity_id: EntityId
    value: object
    version: int
    supersedes_version: int | None
    status: InsightStatus
    created_by_proposal: ProposalId
    valid_from: SimulationTime
    valid_until: SimulationTime | None
    supports: tuple[InsightId, ...] = ()
    contradicts: tuple[InsightId, ...] = ()
    supersedes: tuple[InsightId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.insight_id, InsightId):
            raise TypeError("insight_id must be InsightId")
        if not isinstance(self.entity_id, EntityId):
            raise TypeError("entity_id must be EntityId")
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be positive")
        expected = None if self.version == 1 else self.version - 1
        if self.supersedes_version != expected:
            raise ValueError("supersedes_version must point to the previous version")
        if not isinstance(self.status, InsightStatus):
            raise TypeError("status must be InsightStatus")
        if not isinstance(self.created_by_proposal, ProposalId):
            raise TypeError("created_by_proposal must be ProposalId")
        if not isinstance(self.valid_from, SimulationTime):
            raise TypeError("valid_from must be SimulationTime")
        if self.valid_until is not None:
            if not isinstance(self.valid_until, SimulationTime):
                raise TypeError("valid_until must be SimulationTime or None")
            if self.valid_until.value <= self.valid_from.value:
                raise ValueError("valid_until must be later than valid_from")
        _validate_insight_relations(
            self.insight_id, self.supports, self.contradicts, self.supersedes
        )
        object.__setattr__(self, "value", deepcopy(self.value))


def _validate_insight_relations(
    insight_id: InsightId,
    supports: tuple[InsightId, ...],
    contradicts: tuple[InsightId, ...],
    supersedes: tuple[InsightId, ...],
) -> None:
    groups = (supports, contradicts, supersedes)
    if any(not isinstance(group, tuple) for group in groups):
        raise TypeError("insight relationships must be tuples")
    if any(not isinstance(ref, InsightId) for group in groups for ref in group):
        raise TypeError("insight relationships must contain InsightId values")
    if any(insight_id in group for group in groups):
        raise ValueError("an insight cannot reference itself")
    if any(len(set(group)) != len(group) for group in groups):
        raise ValueError("insight relationships cannot contain duplicates")
    if set(supports) & set(contradicts):
        raise ValueError("an insight cannot both support and contradict the same insight")


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class NewsCategory(str, Enum):
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    CORPORATE_ACTION = "corporate_action"
    MANAGEMENT = "management"
    PRODUCT = "product"
    LEGAL = "legal"
    MACRO = "macro"
    MARKET = "market"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class NewsObservation:
    """A relevance classification, never an investment conclusion."""

    event_id: str
    entities: tuple[str, ...]
    category: NewsCategory
    relevance: float

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if not isinstance(self.entities, tuple) or not self.entities:
            raise ValueError("entities must be a non-empty tuple")
        if any(not isinstance(item, str) for item in self.entities):
            raise TypeError("entities must contain strings")
        normalized = tuple(
            dict.fromkeys(item.strip().upper() for item in self.entities if item.strip())
        )
        if len(normalized) != len(self.entities):
            raise ValueError("entities must be unique non-empty symbols")
        if not isinstance(self.category, NewsCategory):
            raise TypeError("category must be NewsCategory")
        if (
            isinstance(self.relevance, bool)
            or not isinstance(self.relevance, (int, float))
            or not isfinite(float(self.relevance))
            or not 0 <= float(self.relevance) <= 1
        ):
            raise ValueError("relevance must be between 0 and 1")
        object.__setattr__(self, "event_id", self.event_id.strip())
        object.__setattr__(self, "entities", normalized)
        object.__setattr__(self, "relevance", float(self.relevance))


class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class TradeCandidateStatus(str, Enum):
    PROPOSED = "proposed"
    RISK_REVIEWED = "risk_reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TradeCandidate:
    """A proposed action. Thesis lives on the cited insights, not here."""

    id: TradeCandidateId
    instrument: str
    direction: TradeSide
    thesis_refs: tuple[InsightId, ...]
    horizon: str
    confidence: float
    entry_conditions: tuple[str, ...]
    exit_conditions: tuple[str, ...]
    proposed_size: float
    knowledge_time: datetime
    status: TradeCandidateStatus = TradeCandidateStatus.PROPOSED

    def __post_init__(self) -> None:
        if not isinstance(self.id, TradeCandidateId):
            raise TypeError("id must be TradeCandidateId")
        if not isinstance(self.direction, TradeSide):
            raise TypeError("direction must be TradeSide")
        if not isinstance(self.status, TradeCandidateStatus):
            raise TypeError("status must be TradeCandidateStatus")
        instrument = self.instrument.strip().upper() if isinstance(self.instrument, str) else ""
        if not instrument:
            raise ValueError("instrument must be a non-empty string")
        if not isinstance(self.horizon, str) or not self.horizon.strip():
            raise ValueError("horizon must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if isinstance(self.proposed_size, bool) or not isinstance(self.proposed_size, (int, float)):
            raise ValueError("proposed_size must be numeric")
        if not isfinite(float(self.proposed_size)) or self.proposed_size < 0:
            raise ValueError("proposed_size must be a finite non-negative number")
        if self.direction is TradeSide.NO_TRADE:
            if self.proposed_size != 0:
                raise ValueError("no_trade candidates must have proposed_size 0")
        elif self.proposed_size == 0:
            raise ValueError("long and short candidates require a positive proposed_size")
        if not self.thesis_refs:
            raise ValueError("a trade candidate must reference at least one insight")
        if any(not isinstance(ref, InsightId) for ref in self.thesis_refs):
            raise TypeError("thesis_refs must contain only InsightId values")
        for name, values in {
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
        }.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple of strings")
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("knowledge_time must be timezone-aware")
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "horizon", self.horizon.strip())
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "proposed_size", float(self.proposed_size))
        object.__setattr__(self, "thesis_refs", tuple(dict.fromkeys(self.thesis_refs)))


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
class CompanyEntityRecord:
    ticker: str
    exchange: str
    sector: str
    industry: str
    identifiers: Mapping[str, str]
    fundamental_references: tuple[str, ...]
    knowledge_time: datetime

    def __post_init__(self) -> None:
        for name, value in {
            "ticker": self.ticker,
            "exchange": self.exchange,
            "sector": self.sector,
            "industry": self.industry,
        }.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"company entity {name} must be a non-empty string")
        if not isinstance(self.identifiers, Mapping) or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.identifiers.items()
        ):
            raise ValueError("company identifiers must map non-empty strings")
        if not self.identifiers:
            raise ValueError("company entity requires at least one identifier")
        if not self.fundamental_references or any(
            not isinstance(ref, str) or not ref.strip()
            for ref in self.fundamental_references
        ):
            raise ValueError("company entity requires fundamental references")
        if len(set(self.fundamental_references)) != len(
            self.fundamental_references
        ):
            raise ValueError("fundamental references cannot contain duplicates")
        if self.knowledge_time.tzinfo is None:
            raise ValueError("company entity knowledge_time must be timezone-aware")
        object.__setattr__(self, "ticker", self.ticker.strip().upper())
        object.__setattr__(self, "exchange", self.exchange.strip().upper())
        object.__setattr__(self, "sector", self.sector.strip())
        object.__setattr__(self, "industry", self.industry.strip())
        object.__setattr__(
            self,
            "identifiers",
            MappingProxyType(
                {key.strip(): value.strip() for key, value in self.identifiers.items()}
            ),
        )


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
    agent: str
    symbol: str
    company_thesis: str
    bull_case: str
    bear_case: str
    catalysts: tuple[str, ...]
    risks: tuple[str, ...]
    confidence: float
    time_horizon: str
    evidence_refs: tuple[str, ...]
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.agent.strip()
            or not self.symbol.strip()
            or not self.company_thesis.strip()
            or not self.bull_case.strip()
            or not self.bear_case.strip()
            or not self.time_horizon.strip()
        ):
            raise ValueError("company analysis text fields must be non-empty")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("company analysis confidence must be between 0 and 1")
        groups = (
            self.catalysts,
            self.risks,
            self.evidence_refs,
            self.supports,
            self.contradicts,
            self.supersedes,
        )
        if any(not isinstance(group, tuple) for group in groups) or any(
            not isinstance(item, str) or not item.strip()
            for group in groups
            for item in group
        ):
            raise TypeError("company analysis lists must contain non-empty strings")
        if not self.evidence_refs:
            raise ValueError("company analysis requires evidence references")
        relations = (self.supports, self.contradicts, self.supersedes)
        if any(len(set(group)) != len(group) for group in relations):
            raise ValueError("company analysis relationships cannot contain duplicates")
        if set(self.supports) & set(self.contradicts):
            raise ValueError("an insight cannot both support and contradict the same insight")

    def to_insight_revision(self, insight_id: InsightId) -> InsightRevision:
        return InsightRevision(
            insight_id=insight_id,
            value=self,
            supports=tuple(InsightId(ref) for ref in self.supports),
            contradicts=tuple(InsightId(ref) for ref in self.contradicts),
            supersedes=tuple(InsightId(ref) for ref in self.supersedes),
        )


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
    agent: str
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
            "agent": self.agent,
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
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value
