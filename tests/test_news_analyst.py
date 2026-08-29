import unittest
from datetime import datetime, timedelta, timezone

from agents import Evidence, NewsAnalyst
from memory import ContextGateway, ContextPermissionError, NewsEventStore


class StubModel:
    def __init__(self, evidence_refs=None):
        self.evidence_refs = evidence_refs or ["news:1"]
        self.last_input = None

    def generate_json(self, *, instructions, input_data, schema):
        self.last_input = input_data
        return {
            "claim": "Synthetic test finding", "direction": "neutral", "confidence": 0.4,
            "horizon": "test horizon", "evidence_refs": self.evidence_refs,
            "risks": ["Synthetic test uncertainty"],
        }


def article(ref="news:1", symbol="ABC", known_at=None):
    known_at = known_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Evidence(
        ref=ref, source="Synthetic test source", url=f"https://example.test/{ref}",
        published_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Synthetic test headline", summary="Synthetic test summary", symbols=(symbol,),
        knowledge_time=known_at,
    )


def context(*articles, symbol="ABC", at=None):
    shared = NewsEventStore()
    for item in articles:
        shared.append_news(item)
    return ContextGateway(shared).for_news_analyst(
        agent_id="news_analyst", symbol=symbol,
        simulation_time=at or datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


class NewsAnalystTests(unittest.TestCase):
    def test_returns_structured_finding(self):
        model = StubModel()
        finding = NewsAnalyst(model).analyze(context(article()))
        self.assertEqual(finding.subject, "ABC")
        self.assertEqual(finding.evidence_refs, ("news:1",))
        self.assertEqual(model.last_input["articles"][0]["url"], "https://example.test/news:1")

    def test_rejects_hallucinated_evidence_reference(self):
        with self.assertRaises(ValueError):
            NewsAnalyst(StubModel(["news:not-supplied"])).analyze(context(article()))

    def test_requires_sourced_input(self):
        with self.assertRaises(ValueError):
            NewsAnalyst(StubModel()).analyze(context())

    def test_rejects_direct_article_list(self):
        with self.assertRaises(TypeError):
            NewsAnalyst(StubModel()).analyze([article()])

    def test_gateway_excludes_future_and_other_symbols(self):
        simulation_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
        selected = context(
            article("news:visible"),
            article("news:future", known_at=simulation_time + timedelta(days=1)),
            article("news:other", symbol="XYZ"),
            at=simulation_time,
        )
        self.assertEqual([item.ref for item in selected.articles], ["news:visible"])

    def test_context_contains_no_portfolio_state(self):
        selected = context(article())
        self.assertFalse(hasattr(selected, "portfolio"))
        self.assertEqual(set(selected.__slots__), {"symbol", "simulation_time", "articles"})

    def test_gateway_checks_agent_permission(self):
        with self.assertRaises(ContextPermissionError):
            ContextGateway(NewsEventStore()).for_news_analyst(
                agent_id="company_analyst", symbol="ABC",
                simulation_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )

    def test_duplicate_shared_memory_reference_rejected(self):
        shared = NewsEventStore()
        shared.append_news(article())
        with self.assertRaises(ValueError):
            shared.append_news(article())


if __name__ == "__main__":
    unittest.main()
