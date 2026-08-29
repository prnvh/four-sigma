from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.schemas import (
        CompanyAnalysisRecord,
        CompanyRecord,
        Evidence,
        MarketFeature,
        OutcomeDefinition,
        PromotedInsight,
    )


class ContextPermissionError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class NewsAnalystContext:
    """The complete and only context visible to the News Analyst."""

    symbol: str
    simulation_time: datetime
    articles: tuple["Evidence", ...]


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
    records: tuple["CompanyRecord", ...]
    promoted_insights: tuple["PromotedInsight", ...]
    market_features: tuple["MarketFeature", ...]


@dataclass(frozen=True, slots=True)
class RiskAnalystContext:
    """Company and market evidence visible to the AI Risk Analyst."""

    symbol: str
    simulation_time: datetime
    outcome: "OutcomeDefinition"
    company_analyses: tuple["CompanyAnalysisRecord", ...]
    records: tuple["CompanyRecord", ...]
    promoted_insights: tuple["PromotedInsight", ...]
    market_features: tuple["MarketFeature", ...]


class ResearchContextStore:
    """Typed immutable research records used only through ContextGateway."""

    def __init__(self) -> None:
        self._news: dict[str, "Evidence"] = {}
        self._company_records: dict[str, "CompanyRecord"] = {}
        self._promoted_insights: dict[str, "PromotedInsight"] = {}
        self._market_features: dict[str, "MarketFeature"] = {}
        self._company_analyses: dict[str, "CompanyAnalysisRecord"] = {}

    def append_news(self, article: "Evidence") -> None:
        if article.ref in self._news:
            raise ValueError(f"duplicate evidence reference: {article.ref}")
        self._news[article.ref] = article

    def _news_records(self) -> tuple["Evidence", ...]:
        return tuple(self._news.values())

    def append_company_record(self, record: "CompanyRecord") -> None:
        if record.ref in self._company_records:
            raise ValueError(f"duplicate company record reference: {record.ref}")
        self._company_records[record.ref] = record

    def append_promoted_insight(self, insight: "PromotedInsight") -> None:
        if insight.ref in self._promoted_insights:
            raise ValueError(f"duplicate promoted insight reference: {insight.ref}")
        self._promoted_insights[insight.ref] = insight

    def _company_evidence(self) -> tuple["CompanyRecord", ...]:
        return tuple(self._company_records.values())

    def _shared_insights(self) -> tuple["PromotedInsight", ...]:
        return tuple(self._promoted_insights.values())

    def append_market_feature(self, feature: "MarketFeature") -> None:
        if feature.ref in self._market_features:
            raise ValueError(f"duplicate market feature reference: {feature.ref}")
        self._market_features[feature.ref] = feature

    def _market_evidence(self) -> tuple["MarketFeature", ...]:
        return tuple(self._market_features.values())

    def append_company_analysis(self, analysis: "CompanyAnalysisRecord") -> None:
        if analysis.ref in self._company_analyses:
            raise ValueError(f"duplicate company analysis reference: {analysis.ref}")
        self._company_analyses[analysis.ref] = analysis

    def _company_analysis_records(self) -> tuple["CompanyAnalysisRecord", ...]:
        return tuple(self._company_analyses.values())


# Compatibility name retained for the already-merged news analyst.
NewsEventStore = ResearchContextStore


class ContextGateway:
    """Enforces field, entity, time and size boundaries outside the model."""

    NEWS_ANALYST_ID = "news_analyst"
    COMPANY_ANALYST_ID = "company_analyst"
    RISK_ANALYST_ID = "risk_analyst"
    MARKET_AGENT_ID = "market"

    def __init__(
        self,
        shared_memory: ResearchContextStore,
        *,
        max_articles: int = 50,
        max_company_records: int = 100,
        max_promoted_insights: int = 30,
        max_market_features: int = 30,
        max_company_analyses: int = 10,
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

    def for_news_analyst(
        self, *, agent_id: str, symbol: str, simulation_time: datetime
    ) -> NewsAnalystContext:
        if agent_id != self.NEWS_ANALYST_ID:
            raise ContextPermissionError(f"{agent_id!r} cannot request News Analyst context")
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        canonical_symbol = symbol.strip().upper()
        if not canonical_symbol:
            raise ValueError("symbol must be non-empty")

        visible = [
            article
            for article in self._shared_memory._news_records()
            if canonical_symbol in article.symbols
            and article.knowledge_time is not None
            and article.knowledge_time <= simulation_time
        ]
        visible.sort(key=lambda article: article.knowledge_time, reverse=True)
        return NewsAnalystContext(
            symbol=canonical_symbol,
            simulation_time=simulation_time,
            articles=tuple(visible[: self._max_articles]),
        )

    def for_market_agent(
        self, *, agent_id: str, symbols: tuple[str, ...] | list[str], simulation_time: datetime
    ) -> MarketContext:
        if agent_id != self.MARKET_AGENT_ID:
            raise ContextPermissionError(f"{agent_id!r} cannot request Market Agent context")
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        canonical = tuple(
            symbol.strip().upper()
            for symbol in symbols
            if isinstance(symbol, str) and symbol.strip()
        )
        if not canonical:
            raise ValueError("at least one symbol is required")
        return MarketContext(symbols=canonical, simulation_time=simulation_time)

    def for_company_analyst(
        self, *, agent_id: str, symbol: str, simulation_time: datetime
    ) -> CompanyAnalystContext:
        if agent_id != self.COMPANY_ANALYST_ID:
            raise ContextPermissionError(f"{agent_id!r} cannot request Company Analyst context")
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        canonical_symbol = symbol.strip().upper()
        if not canonical_symbol:
            raise ValueError("symbol must be non-empty")

        records = [
            record
            for record in self._shared_memory._company_evidence()
            if record.symbol == canonical_symbol and record.knowledge_time <= simulation_time
        ]
        records.sort(key=lambda record: record.knowledge_time, reverse=True)
        insights = [
            insight
            for insight in self._shared_memory._shared_insights()
            if insight.symbol == canonical_symbol
            and insight.knowledge_time <= simulation_time
            and (insight.valid_until is None or insight.valid_until >= simulation_time)
        ]
        insights.sort(key=lambda insight: insight.knowledge_time, reverse=True)
        features = [
            feature
            for feature in self._shared_memory._market_evidence()
            if feature.symbol == canonical_symbol and feature.knowledge_time <= simulation_time
        ]
        features.sort(key=lambda feature: feature.knowledge_time, reverse=True)
        return CompanyAnalystContext(
            symbol=canonical_symbol,
            simulation_time=simulation_time,
            records=tuple(records[: self._max_company_records]),
            promoted_insights=tuple(insights[: self._max_promoted_insights]),
            market_features=tuple(features[: self._max_market_features]),
        )

    def for_risk_analyst(
        self,
        *,
        agent_id: str,
        symbol: str,
        simulation_time: datetime,
        outcome: "OutcomeDefinition",
    ) -> RiskAnalystContext:
        if agent_id != self.RISK_ANALYST_ID:
            raise ContextPermissionError(f"{agent_id!r} cannot request Risk Analyst context")
        if simulation_time.tzinfo is None:
            raise ValueError("simulation_time must be timezone-aware")
        canonical_symbol = symbol.strip().upper()
        if not canonical_symbol:
            raise ValueError("symbol must be non-empty")

        analyses = [
            item for item in self._shared_memory._company_analysis_records()
            if item.analysis.symbol == canonical_symbol and item.knowledge_time <= simulation_time
        ]
        analyses.sort(key=lambda item: item.knowledge_time, reverse=True)
        records = [
            item for item in self._shared_memory._company_evidence()
            if item.symbol == canonical_symbol and item.knowledge_time <= simulation_time
        ]
        records.sort(key=lambda item: item.knowledge_time, reverse=True)
        insights = [
            item for item in self._shared_memory._shared_insights()
            if item.symbol == canonical_symbol
            and item.knowledge_time <= simulation_time
            and (item.valid_until is None or item.valid_until >= simulation_time)
        ]
        insights.sort(key=lambda item: item.knowledge_time, reverse=True)
        features = [
            item for item in self._shared_memory._market_evidence()
            if item.symbol == canonical_symbol and item.knowledge_time <= simulation_time
        ]
        features.sort(key=lambda item: item.knowledge_time, reverse=True)
        return RiskAnalystContext(
            symbol=canonical_symbol,
            simulation_time=simulation_time,
            outcome=outcome,
            company_analyses=tuple(analyses[: self._max_company_analyses]),
            records=tuple(records[: self._max_company_records]),
            promoted_insights=tuple(insights[: self._max_promoted_insights]),
            market_features=tuple(features[: self._max_market_features]),
        )
