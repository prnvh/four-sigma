import unittest
from datetime import datetime, timezone

from agents import (
    CompanyRecord,
    CompanyRecordType,
    Direction,
    Evidence,
    MarketFeature,
    MarketFeatureType,
    NewsAnalyst,
    PromotedInsight,
)
from memory import (
    AuthorizationError,
    ContextGateway,
    ContextPermissionError,
    ContextPurpose,
    ResearchContextStore,
)
from memory.context_gateway import _content_hash


NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


class StubModel:
    def generate_json(self, *, instructions, input_data, schema):
        return {
            "claim": "Synthetic test finding",
            "direction": "neutral",
            "confidence": 0.4,
            "horizon": "test horizon",
            "evidence_refs": ["news:1"],
            "risks": ["Synthetic test uncertainty"],
        }


def article(ref="news:1", symbol="ABC"):
    return Evidence(
        ref=ref,
        source="Synthetic test source",
        url=f"https://example.test/{ref}",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Synthetic test headline",
        summary="Synthetic test summary",
        symbols=(symbol,),
        knowledge_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def record(ref="company:1", symbol="ABC"):
    return CompanyRecord(
        ref=ref,
        symbol=symbol,
        source="Synthetic filing source",
        url=f"https://example.test/{ref}",
        knowledge_time=NOW,
        record_type=CompanyRecordType.REGULATORY_FILING,
        label="Synthetic filing fact",
        value="Synthetic value",
        period_end="2025-12-31",
    )


def insight(ref="insight:1", symbol="ABC"):
    return PromotedInsight(
        ref=ref,
        symbol=symbol,
        claim="Synthetic promoted claim",
        direction=Direction.NEUTRAL,
        confidence=0.5,
        evidence_refs=("news:1",),
        knowledge_time=NOW,
    )


def feature(ref="market:1", symbol="ABC"):
    return MarketFeature(
        ref=ref,
        symbol=symbol,
        source="Synthetic market source",
        url=f"https://example.test/{ref}",
        knowledge_time=NOW,
        feature=MarketFeatureType.RETURN_20D,
        value=0.05,
        unit="decimal_return",
    )


def store_with_company():
    store = ResearchContextStore()
    store.append_news(article())
    store.append_company_record(record())
    store.append_promoted_insight(insight())
    store.append_market_feature(feature())
    return store


class GetContextTests(unittest.TestCase):
    def test_same_company_yields_different_contexts_by_permission(self) -> None:
        gateway = ContextGateway(store_with_company())
        news = gateway.get_context(
            agent_id="news_analyst",
            purpose=ContextPurpose.NEWS_ANALYSIS,
            entity_ids=("ABC",),
            simulation_time=NOW,
        )
        company = gateway.get_context(
            agent_id="company_analyst",
            purpose=ContextPurpose.COMPANY_ANALYSIS,
            entity_ids=("ABC",),
            simulation_time=NOW,
        )
        self.assertEqual(news.view.symbol, company.view.symbol)
        self.assertTrue(hasattr(news.view, "articles"))
        self.assertFalse(hasattr(news.view, "records"))
        self.assertTrue(hasattr(company.view, "records"))
        self.assertFalse(hasattr(company.view, "articles"))
        self.assertEqual([item.ref for item in news.view.articles], ["news:1"])
        self.assertEqual([item.ref for item in company.view.records], ["company:1"])
        self.assertNotEqual(set(news.snapshot.fields), set(company.snapshot.fields))

    def test_unknown_purpose_and_cross_agent_purpose_are_rejected(self) -> None:
        gateway = ContextGateway(store_with_company())
        with self.assertRaises(ValueError):
            gateway.get_context(
                agent_id="news_analyst",
                purpose="portfolio_peek",
                entity_ids=("ABC",),
                simulation_time=NOW,
            )
        with self.assertRaises(ContextPermissionError):
            gateway.get_context(
                agent_id="news_analyst",
                purpose=ContextPurpose.COMPANY_ANALYSIS,
                entity_ids=("ABC",),
                simulation_time=NOW,
            )
        with self.assertRaises(AuthorizationError):
            gateway.get_context(
                agent_id="news_analyst",
                purpose=ContextPurpose.NEWS_ANALYSIS,
                entity_ids=("ABC",),
                simulation_time=NOW,
                fields=(("portfolio", "positions"),),
            )
        with self.assertRaises(ContextPermissionError):
            gateway.get_context(
                agent_id="news_analyst",
                purpose=ContextPurpose.TRADE_CONSTRUCTION,
                entity_ids=("ABC",),
                simulation_time=NOW,
            )


class ContextSnapshotTests(unittest.TestCase):
    def test_snapshot_records_what_the_agent_saw(self) -> None:
        gateway = ContextGateway(store_with_company())
        bundle = gateway.get_context(
            agent_id="news_analyst",
            purpose="news_analysis",
            entity_ids=("ABC",),
            simulation_time=NOW,
        )
        snapshot = gateway.snapshots.get(bundle.snapshot.id)
        self.assertEqual(snapshot.agent_id, "news_analyst")
        self.assertEqual(snapshot.simulation_time, NOW)
        self.assertEqual(snapshot.fields, (("events", "news"),))
        self.assertEqual(snapshot.source_refs, ("news:1",))
        self.assertEqual(snapshot.content_hash, snapshot.id.value)
        self.assertEqual(gateway.snapshots.view(snapshot.id), bundle.view)

    def test_output_replays_against_stored_snapshot_not_later_memory(self) -> None:
        store = store_with_company()
        gateway = ContextGateway(store)
        bundle = gateway.get_context(
            agent_id="news_analyst",
            purpose=ContextPurpose.NEWS_ANALYSIS,
            entity_ids=("ABC",),
            simulation_time=NOW,
        )
        store.append_news(article(ref="news:later"))
        later = gateway.get_context(
            agent_id="news_analyst",
            purpose=ContextPurpose.NEWS_ANALYSIS,
            entity_ids=("ABC",),
            simulation_time=NOW,
        )
        replayed = gateway.snapshots.view(bundle.snapshot.id)
        first = NewsAnalyst(StubModel()).analyze(bundle.view)
        second = NewsAnalyst(StubModel()).analyze(replayed)
        self.assertEqual(first, second)
        self.assertEqual(replayed, bundle.view)
        self.assertEqual([item.ref for item in replayed.articles], ["news:1"])
        self.assertEqual({item.ref for item in later.view.articles}, {"news:later", "news:1"})
        self.assertNotEqual(later.snapshot.content_hash, bundle.snapshot.content_hash)

    def test_legacy_for_methods_also_persist_snapshots(self) -> None:
        gateway = ContextGateway(store_with_company())
        view = gateway.for_company_analyst(
            agent_id="company_analyst", symbol="ABC", simulation_time=NOW
        )
        snapshot = gateway.snapshots.get(_content_hash(view))
        self.assertEqual(snapshot.agent_id, "company_analyst")
        self.assertEqual(gateway.snapshots.view(snapshot.id), view)
        self.assertIn("company:1", snapshot.source_refs)


if __name__ == "__main__":
    unittest.main()
