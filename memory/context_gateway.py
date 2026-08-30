from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json

from .capabilities import CAPABILITIES, CapabilityModel
from .types import (
    CompanyAnalysisRecord,
    CompanyEntityRecord,
    CompanyRecord,
    ContextSnapshotId,
    Evidence,
    MarketFeature,
    OutcomeDefinition,
    PromotedInsight,
    jsonable,
)


class ContextPermissionError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class NewsAnalystContext:
    """The complete and only context visible to the News Analyst."""

    symbol: str
    simulation_time: datetime
    articles: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class NewsAgentContext:
    """Raw news visible to the News Agent at one simulation timestamp."""

    universe: tuple[str, ...]
    simulation_time: datetime
    articles: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class MarketContext:
    """The complete and only context visible to the Market Agent."""

    symbols: tuple[str, ...]
    simulation_time: datetime


@dataclass(frozen=True, slots=True)
class CompanyAnalystContext:
    """The complete and only context visible to the Company Analyst."""

    symbol: str
    simulation_time: datetime
    company: CompanyEntityRecord | None
    records: tuple[CompanyRecord, ...]
    promoted_insights: tuple[PromotedInsight, ...]
    recent_events: tuple[Evidence, ...]
    market_features: tuple[MarketFeature, ...]


@dataclass(frozen=True, slots=True)
class RiskAnalystContext:
    """Company and market evidence visible to the AI Risk Analyst."""

    symbol: str
    simulation_time: datetime
    outcome: OutcomeDefinition
    company_analyses: tuple[CompanyAnalysisRecord, ...]
    records: tuple[CompanyRecord, ...]
    promoted_insights: tuple[PromotedInsight, ...]
    market_features: tuple[MarketFeature, ...]


@dataclass(frozen=True, slots=True)
class TradeConstructionContext:
    """Approved insights only. No raw news, filings, or portfolio state."""

    symbol: str
    simulation_time: datetime
    promoted_insights: tuple[PromotedInsight, ...]


class ContextPurpose(StrEnum):
    NEWS_INGESTION = "news_ingestion"
    NEWS_ANALYSIS = "news_analysis"
    COMPANY_ANALYSIS = "company_analysis"
    MARKET = "market"
    RISK_ANALYSIS = "risk_analysis"
    TRADE_CONSTRUCTION = "trade_construction"


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Immutable record of exactly the fields and sources an agent saw."""

    id: ContextSnapshotId
    agent_id: str
    simulation_time: datetime
    fields: tuple[tuple[str, str], ...]
    source_refs: tuple[str, ...]
    content_hash: str
    content_size_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.content_size_bytes, bool)
            or not isinstance(self.content_size_bytes, int)
            or self.content_size_bytes < 0
        ):
            raise ValueError("content_size_bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Permissioned view plus the snapshot needed to replay it."""

    view: (
        NewsAgentContext
        | NewsAnalystContext
        | CompanyAnalystContext
        | MarketContext
        | RiskAnalystContext
        | TradeConstructionContext
    )
    snapshot: ContextSnapshot


class ContextSnapshotStore:
    """Append-only snapshot ledger. Replay uses the stored view, not live memory."""

    def __init__(self) -> None:
        self._snapshots: dict[str, ContextSnapshot] = {}
        self._views: dict[str, object] = {}
        self._history: list[ContextSnapshot] = []

    def put(self, snapshot: ContextSnapshot, view: object) -> ContextSnapshot:
        key = snapshot.id.value
        existing = self._snapshots.get(key)
        if existing is not None and existing != snapshot:
            raise ValueError(f"snapshot id collision: {key}")
        self._snapshots[key] = snapshot
        self._views[key] = view
        self._history.append(snapshot)
        return snapshot

    def snapshot(self) -> tuple[ContextSnapshot, ...]:
        """Return context requests in observation order, including repeated views."""

        return tuple(self._history)

    def get(self, snapshot_id: ContextSnapshotId | str) -> ContextSnapshot:
        key = snapshot_id.value if isinstance(snapshot_id, ContextSnapshotId) else snapshot_id
        snapshot = self._snapshots.get(key)
        if snapshot is None:
            raise KeyError(f"unknown context snapshot: {key}")
        return snapshot

    def view(self, snapshot_id: ContextSnapshotId | str) -> object:
        key = snapshot_id.value if isinstance(snapshot_id, ContextSnapshotId) else snapshot_id
        if key not in self._views:
            raise KeyError(f"unknown context snapshot: {key}")
        return self._views[key]


def _canonical_entities(entity_ids: Sequence[str]) -> tuple[str, ...]:
    entities = tuple(
        item.strip().upper()
        for item in entity_ids
        if isinstance(item, str) and item.strip()
    )
    if not entities:
        raise ValueError("at least one entity id is required")
    return entities


def _source_refs(view: object) -> tuple[str, ...]:
    refs: list[str] = []
    for name in (
        "articles",
        "records",
        "promoted_insights",
        "recent_events",
        "market_features",
        "company_analyses",
    ):
        for item in getattr(view, name, ()):
            ref = getattr(item, "ref", None)
            if isinstance(ref, str) and ref:
                refs.append(ref)
    return tuple(dict.fromkeys(refs))


def _content_hash(view: object) -> str:
    blob = _content_blob(view)
    return sha256(blob.encode("utf-8")).hexdigest()


def _content_blob(view: object) -> str:
    return json.dumps(
        jsonable(view), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


class ResearchContextStore:
    """Typed immutable research records used only through ContextGateway."""

    def __init__(self) -> None:
        self._news: dict[str, Evidence] = {}
        self._company_entities: list[CompanyEntityRecord] = []
        self._company_records: dict[str, CompanyRecord] = {}
        self._promoted_insights: dict[str, PromotedInsight] = {}
        self._market_features: dict[str, MarketFeature] = {}
        self._company_analyses: dict[str, CompanyAnalysisRecord] = {}

    def append_news(self, article: Evidence) -> None:
        if article.ref in self._news:
            raise ValueError(f"duplicate evidence reference: {article.ref}")
        self._news[article.ref] = article

    def _news_records(self) -> tuple[Evidence, ...]:
        return tuple(self._news.values())

    def append_company_entity(self, company: CompanyEntityRecord) -> None:
        if not isinstance(company, CompanyEntityRecord):
            raise TypeError("company must be CompanyEntityRecord")
        missing = set(company.fundamental_references) - set(self._company_records)
        if missing:
            raise ValueError(
                f"unknown fundamental references: {sorted(missing)}"
            )
        facts = [
            self._company_records[ref]
            for ref in company.fundamental_references
        ]
        if any(fact.symbol != company.ticker for fact in facts):
            raise ValueError("fundamental references must belong to the company ticker")
        if any(fact.knowledge_time > company.knowledge_time for fact in facts):
            raise ValueError("company entity cannot reference future fundamentals")
        key = (company.ticker, company.exchange, company.knowledge_time)
        if any(
            (item.ticker, item.exchange, item.knowledge_time) == key
            for item in self._company_entities
        ):
            raise ValueError("duplicate company entity record")
        self._company_entities.append(company)

    def _company_entity_records(self) -> tuple[CompanyEntityRecord, ...]:
        return tuple(self._company_entities)

    def append_company_record(self, record: CompanyRecord) -> None:
        if record.ref in self._company_records:
            raise ValueError(f"duplicate company record reference: {record.ref}")
        self._company_records[record.ref] = record

    def append_promoted_insight(self, insight: PromotedInsight) -> None:
        if insight.ref in self._promoted_insights:
            raise ValueError(f"duplicate promoted insight reference: {insight.ref}")
        self._promoted_insights[insight.ref] = insight

    def _company_evidence(self) -> tuple[CompanyRecord, ...]:
        return tuple(self._company_records.values())

    def _shared_insights(self) -> tuple[PromotedInsight, ...]:
        return tuple(self._promoted_insights.values())

    def append_market_feature(self, feature: MarketFeature) -> None:
        if feature.ref in self._market_features:
            raise ValueError(f"duplicate market feature reference: {feature.ref}")
        self._market_features[feature.ref] = feature

    def _market_evidence(self) -> tuple[MarketFeature, ...]:
        return tuple(self._market_features.values())

    def append_company_analysis(self, analysis: CompanyAnalysisRecord) -> None:
        if analysis.ref in self._company_analyses:
            raise ValueError(f"duplicate company analysis reference: {analysis.ref}")
        self._company_analyses[analysis.ref] = analysis

    def _company_analysis_records(self) -> tuple[CompanyAnalysisRecord, ...]:
        return tuple(self._company_analyses.values())


# Compatibility name retained for the already-merged news analyst.
NewsEventStore = ResearchContextStore


class ContextGateway:
    """Enforces field, entity, time and size boundaries outside the model."""

    NEWS_ANALYST_ID = "news_analyst"
    COMPANY_ANALYST_ID = "company_analyst"
    RISK_ANALYST_ID = "risk_analyst"
    MARKET_AGENT_ID = "market"
    TRADE_CONSTRUCTOR_ID = "trade_constructor"

    def __init__(
        self,
        shared_memory: ResearchContextStore,
        *,
        max_articles: int = 50,
        max_company_records: int = 100,
        max_promoted_insights: int = 30,
        max_market_features: int = 30,
        max_company_analyses: int = 10,
        capabilities: CapabilityModel | None = None,
        snapshots: ContextSnapshotStore | None = None,
    ) -> None:
        if min(
            max_articles, max_company_records, max_promoted_insights,
            max_market_features, max_company_analyses,
        ) < 1:
            raise ValueError("context limits must be positive")
        self._shared_memory = shared_memory
        self._max_articles = max_articles
        self._max_company_records = max_company_records
        self._max_promoted_insights = max_promoted_insights
        self._max_market_features = max_market_features
        self._max_company_analyses = max_company_analyses
        self._capabilities = capabilities or CAPABILITIES
        self.snapshots = snapshots or ContextSnapshotStore()

    def _require_reads(
        self, agent_id: str, fields: Sequence[tuple[str, str]]
    ) -> None:
        self._capabilities.require_reads(agent_id, fields)

    def _record(
        self,
        *,
        agent_id: str,
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]],
        view: object,
    ) -> ContextSnapshot:
        digest = _content_hash(view)
        snapshot = ContextSnapshot(
            id=ContextSnapshotId(digest),
            agent_id=agent_id,
            simulation_time=simulation_time,
            fields=tuple(fields),
            source_refs=_source_refs(view),
            content_hash=digest,
            content_size_bytes=len(_content_blob(view).encode("utf-8")),
        )
        return self.snapshots.put(snapshot, view)

    def get_context(
        self,
        *,
        agent_id: str,
        purpose: ContextPurpose | str,
        entity_ids: Sequence[str],
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
        outcome: OutcomeDefinition | None = None,
    ) -> ContextBundle:
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        try:
            selected = purpose if isinstance(purpose, ContextPurpose) else ContextPurpose(purpose)
        except ValueError as error:
            raise ValueError(f"unknown context purpose: {purpose!r}") from error
        entities = _canonical_entities(entity_ids)
        view: (
            NewsAgentContext
            | NewsAnalystContext
            | CompanyAnalystContext
            | MarketContext
            | RiskAnalystContext
            | TradeConstructionContext
        )
        granted: tuple[tuple[str, str], ...]
        if selected is ContextPurpose.NEWS_INGESTION:
            if agent_id != "news":
                raise ContextPermissionError(f"{agent_id!r} cannot request News Agent context")
            granted = (("events", "news"), *fields)
            self._require_reads(agent_id, granted)
            view = self._news_agent_view(entities, simulation_time)
        elif selected is ContextPurpose.NEWS_ANALYSIS:
            if agent_id != self.NEWS_ANALYST_ID:
                raise ContextPermissionError(f"{agent_id!r} cannot request News Analyst context")
            if len(entities) != 1:
                raise ValueError("news_analysis requires exactly one entity")
            granted = (("events", "news"), *fields)
            self._require_reads(agent_id, granted)
            view = self._news_view(entities[0], simulation_time)
        elif selected is ContextPurpose.COMPANY_ANALYSIS:
            if agent_id != self.COMPANY_ANALYST_ID:
                raise ContextPermissionError(f"{agent_id!r} cannot request Company Analyst context")
            if len(entities) != 1:
                raise ValueError("company_analysis requires exactly one entity")
            granted = (
                ("company", "entity"),
                ("company", "records"),
                ("insights", "promoted"),
                ("events", "news"),
                ("market", "features"),
                *fields,
            )
            self._require_reads(agent_id, granted)
            view = self._company_view(entities[0], simulation_time)
        elif selected is ContextPurpose.MARKET:
            if agent_id != self.MARKET_AGENT_ID:
                raise ContextPermissionError(f"{agent_id!r} cannot request Market Agent context")
            granted = (("market", "ohlcv"), *fields)
            self._require_reads(agent_id, granted)
            view = MarketContext(symbols=entities, simulation_time=simulation_time)
        elif selected is ContextPurpose.RISK_ANALYSIS:
            if agent_id != self.RISK_ANALYST_ID:
                raise ContextPermissionError(f"{agent_id!r} cannot request Risk Analyst context")
            if outcome is None:
                raise ValueError("risk_analysis requires an outcome definition")
            if len(entities) != 1:
                raise ValueError("risk_analysis requires exactly one entity")
            granted = (
                ("company", "records"),
                ("company", "analyses"),
                ("insights", "promoted"),
                ("market", "features"),
                *fields,
            )
            self._require_reads(agent_id, granted)
            view = self._risk_view(entities[0], simulation_time, outcome)
        elif selected is ContextPurpose.TRADE_CONSTRUCTION:
            if agent_id != self.TRADE_CONSTRUCTOR_ID:
                raise ContextPermissionError(
                    f"{agent_id!r} cannot request trade-construction context"
                )
            if len(entities) != 1:
                raise ValueError("trade_construction requires exactly one entity")
            granted = (("insights", "promoted"), *fields)
            self._require_reads(agent_id, granted)
            view = self._trade_view(entities[0], simulation_time)
        else:
            raise ValueError(f"unknown context purpose: {purpose!r}")
        return ContextBundle(
            view=view,
            snapshot=self._record(
                agent_id=agent_id,
                simulation_time=simulation_time,
                fields=granted,
                view=view,
            ),
        )

    def _news_view(self, symbol: str, simulation_time: datetime) -> NewsAnalystContext:
        visible = [
            article
            for article in self._shared_memory._news_records()
            if symbol in article.symbols
            and article.knowledge_time is not None
            and article.knowledge_time <= simulation_time
        ]
        visible.sort(key=lambda article: article.knowledge_time, reverse=True)
        return NewsAnalystContext(
            symbol=symbol,
            simulation_time=simulation_time,
            articles=tuple(visible[: self._max_articles]),
        )

    def _news_agent_view(
        self, universe: tuple[str, ...], simulation_time: datetime
    ) -> NewsAgentContext:
        allowed = set(universe)
        visible = [
            article
            for article in self._shared_memory._news_records()
            if article.knowledge_time is not None
            and article.knowledge_time <= simulation_time
            and (not article.symbols or allowed.intersection(article.symbols))
        ]
        visible.sort(key=lambda article: (article.knowledge_time, article.ref), reverse=True)
        return NewsAgentContext(
            universe=universe,
            simulation_time=simulation_time,
            articles=tuple(visible[: self._max_articles]),
        )

    def _company_view(self, symbol: str, simulation_time: datetime) -> CompanyAnalystContext:
        companies = [
            company
            for company in self._shared_memory._company_entity_records()
            if company.ticker == symbol and company.knowledge_time <= simulation_time
        ]
        companies.sort(key=lambda company: company.knowledge_time, reverse=True)
        records = [
            record
            for record in self._shared_memory._company_evidence()
            if record.symbol == symbol and record.knowledge_time <= simulation_time
        ]
        records.sort(key=lambda record: record.knowledge_time, reverse=True)
        insights = self._visible_insights(symbol, simulation_time)
        events = [
            event
            for event in self._shared_memory._news_records()
            if symbol in event.symbols
            and event.knowledge_time is not None
            and event.knowledge_time <= simulation_time
        ]
        events.sort(key=lambda event: event.knowledge_time, reverse=True)
        features = [
            feature
            for feature in self._shared_memory._market_evidence()
            if feature.symbol == symbol and feature.knowledge_time <= simulation_time
        ]
        features.sort(key=lambda feature: feature.knowledge_time, reverse=True)
        return CompanyAnalystContext(
            symbol=symbol,
            simulation_time=simulation_time,
            company=companies[0] if companies else None,
            records=tuple(records[: self._max_company_records]),
            promoted_insights=tuple(insights[: self._max_promoted_insights]),
            recent_events=tuple(events[: self._max_articles]),
            market_features=tuple(features[: self._max_market_features]),
        )

    def _risk_view(
        self, symbol: str, simulation_time: datetime, outcome: OutcomeDefinition
    ) -> RiskAnalystContext:
        analyses = [
            item for item in self._shared_memory._company_analysis_records()
            if item.analysis.symbol == symbol and item.knowledge_time <= simulation_time
        ]
        analyses.sort(key=lambda item: item.knowledge_time, reverse=True)
        company = self._company_view(symbol, simulation_time)
        return RiskAnalystContext(
            symbol=symbol,
            simulation_time=simulation_time,
            outcome=outcome,
            company_analyses=tuple(analyses[: self._max_company_analyses]),
            records=company.records,
            promoted_insights=company.promoted_insights,
            market_features=company.market_features,
        )

    def _visible_insights(
        self, symbol: str, simulation_time: datetime
    ) -> list[PromotedInsight]:
        insights = [
            insight
            for insight in self._shared_memory._shared_insights()
            if insight.symbol == symbol
            and insight.knowledge_time <= simulation_time
            and (insight.valid_until is None or insight.valid_until >= simulation_time)
        ]
        insights.sort(key=lambda insight: insight.knowledge_time, reverse=True)
        return insights

    def _trade_view(self, symbol: str, simulation_time: datetime) -> TradeConstructionContext:
        insights = self._visible_insights(symbol, simulation_time)
        return TradeConstructionContext(
            symbol=symbol,
            simulation_time=simulation_time,
            promoted_insights=tuple(insights[: self._max_promoted_insights]),
        )

    def for_news_analyst(
        self,
        *,
        agent_id: str,
        symbol: str,
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
    ) -> NewsAnalystContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.NEWS_ANALYSIS,
            entity_ids=(symbol,),
            simulation_time=simulation_time,
            fields=fields,
        ).view
        assert isinstance(view, NewsAnalystContext)
        return view

    def for_news_agent(
        self,
        *,
        agent_id: str,
        universe: tuple[str, ...] | list[str],
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
    ) -> NewsAgentContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.NEWS_INGESTION,
            entity_ids=universe,
            simulation_time=simulation_time,
            fields=fields,
        ).view
        assert isinstance(view, NewsAgentContext)
        return view

    def for_market_agent(
        self,
        *,
        agent_id: str,
        symbols: tuple[str, ...] | list[str],
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
    ) -> MarketContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.MARKET,
            entity_ids=symbols,
            simulation_time=simulation_time,
            fields=fields,
        ).view
        assert isinstance(view, MarketContext)
        return view

    def for_company_analyst(
        self,
        *,
        agent_id: str,
        symbol: str,
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
    ) -> CompanyAnalystContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.COMPANY_ANALYSIS,
            entity_ids=(symbol,),
            simulation_time=simulation_time,
            fields=fields,
        ).view
        assert isinstance(view, CompanyAnalystContext)
        return view

    def for_risk_analyst(
        self,
        *,
        agent_id: str,
        symbol: str,
        simulation_time: datetime,
        outcome: OutcomeDefinition,
        fields: Sequence[tuple[str, str]] = (),
    ) -> RiskAnalystContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.RISK_ANALYSIS,
            entity_ids=(symbol,),
            simulation_time=simulation_time,
            fields=fields,
            outcome=outcome,
        ).view
        assert isinstance(view, RiskAnalystContext)
        return view

    def for_trade_constructor(
        self,
        *,
        agent_id: str,
        symbol: str,
        simulation_time: datetime,
        fields: Sequence[tuple[str, str]] = (),
    ) -> TradeConstructionContext:
        view = self.get_context(
            agent_id=agent_id,
            purpose=ContextPurpose.TRADE_CONSTRUCTION,
            entity_ids=(symbol,),
            simulation_time=simulation_time,
            fields=fields,
        ).view
        assert isinstance(view, TradeConstructionContext)
        return view
