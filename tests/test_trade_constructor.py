import unittest
from datetime import datetime, timezone

from agents import TRADE_CONSTRUCTOR_V1, TradeConstructor
from memory import (
    AuthorizationError,
    ContextGateway,
    ContextPermissionError,
    ContextPurpose,
    Direction,
    InsightId,
    PromotedInsight,
    ResearchContextStore,
    TradeConstructionContext,
    TradeSide,
)
from memory.context_gateway import NewsAnalystContext
from memory.types import Evidence


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def insight(
    ref: str,
    *,
    direction: Direction = Direction.BULLISH,
    confidence: float = 0.7,
    symbol: str = "ABC",
) -> PromotedInsight:
    return PromotedInsight(
        ref=ref,
        symbol=symbol,
        claim=f"Synthetic claim {ref}",
        direction=direction,
        confidence=confidence,
        evidence_refs=("news:1",),
        knowledge_time=NOW,
    )


def article() -> Evidence:
    return Evidence(
        ref="news:1",
        source="Synthetic test source",
        url="https://example.test/news:1",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Synthetic test headline",
        summary="Synthetic test summary",
        symbols=("ABC",),
        knowledge_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def store(*insights: PromotedInsight) -> ResearchContextStore:
    shared = ResearchContextStore()
    shared.append_news(article())
    for item in insights:
        shared.append_promoted_insight(item)
    return shared


def trade_context(*insights: PromotedInsight) -> TradeConstructionContext:
    return ContextGateway(store(*insights)).for_trade_constructor(
        agent_id="trade_constructor",
        symbol="ABC",
        simulation_time=NOW,
    )


class TradeConstructorTests(unittest.TestCase):
    def test_maps_approved_bullish_majority_to_long_and_cites_them(self) -> None:
        candidate = TradeConstructor().propose(
            trade_context(
                insight("insight:bull-1"),
                insight("insight:bull-2"),
                insight("insight:bear-1", direction=Direction.BEARISH),
            )
        )
        self.assertEqual(candidate.direction, TradeSide.LONG)
        self.assertEqual(
            candidate.thesis_refs,
            (InsightId("insight:bull-1"), InsightId("insight:bull-2")),
        )
        self.assertGreater(candidate.proposed_size, 0)
        self.assertEqual(candidate.horizon, TRADE_CONSTRUCTOR_V1.config["horizon"])

    def test_maps_bearish_majority_to_short(self) -> None:
        candidate = TradeConstructor().propose(
            trade_context(
                insight("insight:bear-1", direction=Direction.BEARISH),
                insight("insight:bear-2", direction=Direction.BEARISH, confidence=0.8),
                insight("insight:bull-1"),
            )
        )
        self.assertEqual(candidate.direction, TradeSide.SHORT)
        self.assertEqual(
            candidate.thesis_refs,
            (InsightId("insight:bear-1"), InsightId("insight:bear-2")),
        )

    def test_conflict_and_neutral_yield_no_trade_citing_all_insights(self) -> None:
        conflict = TradeConstructor().propose(
            trade_context(
                insight("insight:bull-1"),
                insight("insight:bear-1", direction=Direction.BEARISH),
            )
        )
        self.assertEqual(conflict.direction, TradeSide.NO_TRADE)
        self.assertEqual(conflict.proposed_size, 0)
        self.assertEqual(
            set(conflict.thesis_refs),
            {InsightId("insight:bull-1"), InsightId("insight:bear-1")},
        )
        sit_out = TradeConstructor().propose(
            trade_context(insight("insight:flat", direction=Direction.NEUTRAL))
        )
        self.assertEqual(sit_out.direction, TradeSide.NO_TRADE)
        self.assertEqual(sit_out.thesis_refs, (InsightId("insight:flat"),))

    def test_refuses_empty_insights_and_raw_news_context(self) -> None:
        empty = TradeConstructionContext(
            symbol="ABC", simulation_time=NOW, promoted_insights=()
        )
        with self.assertRaises(ValueError):
            TradeConstructor().propose(empty)
        news = NewsAnalystContext(symbol="ABC", simulation_time=NOW, articles=(article(),))
        with self.assertRaises(TypeError):
            TradeConstructor().propose(news)  # type: ignore[arg-type]

    def test_news_cannot_become_a_buy(self) -> None:
        self.assertFalse(hasattr(trade_context(insight("insight:1")), "articles"))
        gateway = ContextGateway(store(insight("insight:1")))
        with self.assertRaises(ContextPermissionError):
            gateway.get_context(
                agent_id="news_analyst",
                purpose=ContextPurpose.TRADE_CONSTRUCTION,
                entity_ids=("ABC",),
                simulation_time=NOW,
            )
        with self.assertRaises(ContextPermissionError):
            gateway.for_trade_constructor(
                agent_id="news_analyst",
                symbol="ABC",
                simulation_time=NOW,
            )

    def test_cannot_request_news_or_portfolio_fields(self) -> None:
        gateway = ContextGateway(store(insight("insight:1")))
        with self.assertRaises(AuthorizationError):
            gateway.for_trade_constructor(
                agent_id="trade_constructor",
                symbol="ABC",
                simulation_time=NOW,
                fields=(("events", "news"),),
            )
        with self.assertRaises(AuthorizationError):
            TradeConstructor().propose(
                trade_context(insight("insight:1")),
                requested_fields=(("portfolio", "positions"),),
            )

    def test_trade_context_excludes_raw_news_even_when_store_has_articles(self) -> None:
        view = trade_context(insight("insight:1"))
        self.assertEqual([item.ref for item in view.promoted_insights], ["insight:1"])
        self.assertFalse(hasattr(view, "articles"))
        self.assertFalse(hasattr(view, "recent_events"))
        self.assertFalse(hasattr(view, "records"))


if __name__ == "__main__":
    unittest.main()
