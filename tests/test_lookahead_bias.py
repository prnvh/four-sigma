import unittest
from datetime import datetime, timezone

from memory import (
    CompanyRecord,
    CompanyRecordType,
    ContextGateway,
    Direction,
    Evidence,
    HistoricalAdapter,
    MarketTape,
    PricePrint,
    PromotedInsight,
    ResearchContextStore,
)


DAY1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
DAY2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
DAY3 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def adapter(store=None, prints=()):
    return HistoricalAdapter(store or ResearchContextStore(), MarketTape(prints))


def company_record(ref, value, known_at, record_type=CompanyRecordType.FINANCIAL_FACT):
    return CompanyRecord(
        ref=ref,
        symbol="ABC",
        source="Point-in-time fixture",
        url=f"https://example.test/{ref}",
        knowledge_time=known_at,
        record_type=record_type,
        label="reported revenue",
        value=value,
        period_end="2025-12-31",
    )


class LookaheadBiasTests(unittest.TestCase):
    def test_future_news_is_hidden_by_knowledge_time_not_publication_time(self):
        store = ResearchContextStore()
        store.append_news(Evidence(
            ref="news:delayed",
            source="Delayed fixture",
            url="https://example.test/news-delayed",
            published_at=DAY1,
            knowledge_time=DAY3,
            title="Old publication ingested later",
            symbols=("ABC",),
        ))

        self.assertEqual(adapter(store).events_as_of(DAY2), ())
        self.assertEqual(
            tuple(item.ref for item in adapter(store).events_as_of(DAY3)),
            ("news:delayed",),
        )

    def test_earnings_period_end_does_not_reveal_unpublished_results(self):
        store = ResearchContextStore()
        earnings = company_record(
            "earnings:2025", "1200000", DAY3, CompanyRecordType.EARNINGS_RELEASE
        )
        store.append_company_record(earnings)

        self.assertEqual(adapter(store).company_data_as_of(DAY2), ())
        self.assertEqual(adapter(store).company_data_as_of(DAY3), (earnings,))

    def test_future_fundamental_revision_cannot_replace_known_value_early(self):
        store = ResearchContextStore()
        original = company_record("fundamental:original", "100", DAY1)
        revision = company_record("fundamental:revision", "92", DAY3)
        store.append_company_record(original)
        store.append_company_record(revision)

        self.assertEqual(adapter(store).company_data_as_of(DAY2), (original,))
        self.assertEqual(
            adapter(store).company_data_as_of(DAY3), (original, revision)
        )

    def test_future_price_is_unavailable_until_its_market_knowledge_time(self):
        tape = (
            PricePrint(symbol="ABC", price=10, knowledge_time=DAY1),
            PricePrint(symbol="ABC", price=50, knowledge_time=DAY3),
        )

        self.assertEqual(adapter(prints=tape).prices_as_of(DAY2), {"ABC": 10})
        self.assertEqual(adapter(prints=tape).prices_as_of(DAY3), {"ABC": 50})

    def test_future_shared_memory_insight_cannot_reach_trade_constructor(self):
        store = ResearchContextStore()
        future = PromotedInsight(
            ref="insight:future",
            symbol="ABC",
            claim="Future-only claim",
            direction=Direction.BULLISH,
            confidence=0.8,
            evidence_refs=("news:future",),
            knowledge_time=DAY3,
        )
        store.append_promoted_insight(future)
        gateway = ContextGateway(store)

        before = gateway.for_trade_constructor(
            agent_id="trade_constructor", symbol="ABC", simulation_time=DAY2
        )
        visible = gateway.for_trade_constructor(
            agent_id="trade_constructor", symbol="ABC", simulation_time=DAY3
        )
        self.assertEqual(before.promoted_insights, ())
        self.assertEqual(visible.promoted_insights, (future,))

    def test_expired_shared_memory_insight_cannot_reappear(self):
        store = ResearchContextStore()
        expired = PromotedInsight(
            ref="insight:expired",
            symbol="ABC",
            claim="Expired claim",
            direction=Direction.NEUTRAL,
            confidence=0.5,
            evidence_refs=("news:old",),
            knowledge_time=DAY1,
            valid_until=DAY2,
        )
        store.append_promoted_insight(expired)

        view = ContextGateway(store).for_trade_constructor(
            agent_id="trade_constructor", symbol="ABC", simulation_time=DAY3
        )
        self.assertEqual(view.promoted_insights, ())


if __name__ == "__main__":
    unittest.main()
