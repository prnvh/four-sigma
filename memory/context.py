from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.schemas import Evidence


class ContextPermissionError(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class NewsAnalystContext:
    """The complete and only context visible to the News Analyst."""

    symbol: str
    simulation_time: datetime
    articles: tuple["Evidence", ...]


class SharedMemory:
    """Immutable news records indexed by evidence reference."""

    def __init__(self) -> None:
        self._news: dict[str, "Evidence"] = {}

    def append_news(self, article: "Evidence") -> None:
        if article.ref in self._news:
            raise ValueError(f"duplicate evidence reference: {article.ref}")
        self._news[article.ref] = article

    def _news_records(self) -> tuple["Evidence", ...]:
        return tuple(self._news.values())


class ContextGateway:
    """Enforces field, entity, time and size boundaries outside the model."""

    NEWS_ANALYST_ID = "news_analyst"

    def __init__(self, shared_memory: SharedMemory, *, max_articles: int = 50) -> None:
        if max_articles < 1:
            raise ValueError("max_articles must be positive")
        self._shared_memory = shared_memory
        self._max_articles = max_articles

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
