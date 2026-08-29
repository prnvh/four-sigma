from __future__ import annotations

from datetime import datetime

from .context_gateway import ResearchContextStore
from .execution import MarketTape
from .portfolio import PortfolioSnapshot
from .types import CompanyRecord, Evidence


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


class HistoricalAdapter:
    """Point-in-time reads. Nothing after knowledge_time is visible."""

    def __init__(
        self,
        store: ResearchContextStore,
        tape: MarketTape,
        snapshots: tuple[PortfolioSnapshot, ...] = (),
    ) -> None:
        if not isinstance(store, ResearchContextStore):
            raise TypeError("store must be ResearchContextStore")
        if not isinstance(tape, MarketTape):
            raise TypeError("tape must be MarketTape")
        self._store = store
        self._tape = tape
        self._snapshots = snapshots

    def prices_as_of(self, when: datetime) -> dict[str, float]:
        return self._tape.prices_as_of(_aware(when, "when"))

    def events_as_of(self, when: datetime) -> tuple[Evidence, ...]:
        when = _aware(when, "when")
        visible = [
            article
            for article in self._store._news_records()
            if article.knowledge_time is not None and article.knowledge_time <= when
        ]
        visible.sort(key=lambda article: article.knowledge_time)
        return tuple(visible)

    def company_data_as_of(self, when: datetime) -> tuple[CompanyRecord, ...]:
        when = _aware(when, "when")
        visible = [
            record
            for record in self._store._company_evidence()
            if record.knowledge_time <= when
        ]
        visible.sort(key=lambda record: record.knowledge_time)
        return tuple(visible)

    def portfolio_as_of(self, when: datetime) -> PortfolioSnapshot | None:
        when = _aware(when, "when")
        visible = [item for item in self._snapshots if item.knowledge_time <= when]
        if not visible:
            return None
        return max(visible, key=lambda item: item.knowledge_time)
