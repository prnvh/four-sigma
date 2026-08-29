import unittest
from datetime import datetime, timedelta, timezone

from agents import NewsAgent, NewsCategory
from memory import ContextGateway, ContextPermissionError, Evidence, NewsEventStore


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


class StubModel:
    def __init__(self, response=None):
        self.response = response
        self.last_input = None
        self.calls = 0

    def generate_json(self, *, instructions, input_data, schema):
        self.calls += 1
        self.last_input = input_data
        if self.response is not None:
            return self.response
        event = input_data["articles"][0]
        return {
            "observations": [{
                "event_id": event["ref"],
                "entities": [event["symbols"][0]],
                "category": "earnings",
                "relevance": 0.85,
            }]
        }


def article(
    ref="news:1",
    *,
    symbol="ABC",
    known_at=NOW,
    url=None,
    title="Synthetic earnings headline",
):
    symbols = () if symbol is None else (symbol,)
    return Evidence(
        ref=ref,
        source="Synthetic test source",
        url=url or f"https://example.test/{ref}",
        published_at=known_at,
        knowledge_time=known_at,
        title=title,
        summary="Synthetic news summary",
        symbols=symbols,
    )


def context(*articles, universe=("ABC",), at=NOW):
    store = NewsEventStore()
    for item in articles:
        store.append_news(item)
    return ContextGateway(store).for_news_agent(
        agent_id="news", universe=universe, simulation_time=at
    )


class NewsAgentTests(unittest.TestCase):
    def test_returns_only_structured_observations(self):
        result = NewsAgent(StubModel()).observe(context(article()))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_id, "news:1")
        self.assertEqual(result[0].entities, ("ABC",))
        self.assertEqual(result[0].category, NewsCategory.EARNINGS)
        self.assertEqual(result[0].relevance, 0.85)
        self.assertEqual(
            set(result[0].__slots__), {"event_id", "entities", "category", "relevance"}
        )

    def test_gateway_excludes_future_and_irrelevant_tagged_news(self):
        selected = context(
            article("news:visible"),
            article("news:future", known_at=NOW + timedelta(minutes=1)),
            article("news:other", symbol="XYZ"),
            article("news:untagged", symbol=None),
        )
        self.assertEqual(
            {item.ref for item in selected.articles}, {"news:visible", "news:untagged"}
        )

    def test_deduplicates_same_url_or_normalized_headline_before_model(self):
        model = StubModel()
        selected = context(
            article("news:z", title="Company Raises Guidance!"),
            article(
                "news:y",
                url="https://example.test/news:z/",
                title="Different syndicated headline",
            ),
            article("news:x", title="Company raises guidance"),
        )
        NewsAgent(model).observe(selected)
        self.assertEqual(len(model.last_input["articles"]), 1)
        self.assertEqual(model.last_input["articles"][0]["ref"], "news:z")

    def test_empty_context_does_not_call_model(self):
        model = StubModel()
        self.assertEqual(NewsAgent(model).observe(context()), ())
        self.assertEqual(model.calls, 0)

    def test_rejects_unknown_event_reference(self):
        model = StubModel({"observations": [{
            "event_id": "news:invented", "entities": ["ABC"],
            "category": "other", "relevance": 0.2,
        }]})
        with self.assertRaisesRegex(ValueError, "unknown event"):
            NewsAgent(model).observe(context(article()))

    def test_rejects_unknown_or_contradictory_entity(self):
        model = StubModel({"observations": [{
            "event_id": "news:1", "entities": ["XYZ"],
            "category": "other", "relevance": 0.2,
        }]})
        with self.assertRaisesRegex(ValueError, "contradicts supplied entity tags"):
            NewsAgent(model).observe(context(article(), universe=("ABC", "XYZ")))

    def test_rejects_investment_conclusion_field(self):
        model = StubModel({"observations": [{
            "event_id": "news:1", "entities": ["ABC"],
            "category": "other", "relevance": 0.2, "direction": "bullish",
        }]})
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            NewsAgent(model).observe(context(article()))

    def test_context_is_permissioned_and_contains_no_investment_state(self):
        selected = context(article())
        self.assertEqual(
            set(selected.__slots__), {"universe", "simulation_time", "articles"}
        )
        with self.assertRaises(ContextPermissionError):
            ContextGateway(NewsEventStore()).for_news_agent(
                agent_id="company_analyst", universe=("ABC",), simulation_time=NOW
            )


if __name__ == "__main__":
    unittest.main()
