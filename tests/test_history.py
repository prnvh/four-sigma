import unittest
from datetime import datetime, timezone

from agents.history_feed import load_equity_news, load_equity_tape
from memory import (
    Evidence,
    HistoricalAdapter,
    MarketTape,
    PricePrint,
    ResearchContextStore,
)


DAY1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
DAY2 = datetime(2026, 3, 2, tzinfo=timezone.utc)
DAY3 = datetime(2026, 3, 3, tzinfo=timezone.utc)


def _yahoo(close_times: tuple[tuple[datetime, float], ...], name: str = "Apple Inc."):
    stamps = []
    closes = []
    for known, price in close_times:
        stamps.append(int((known.timestamp() - 15 * 60)))
        closes.append(price)
    return {
        "chart": {
            "result": [
                {
                    "meta": {"shortName": name},
                    "timestamp": stamps,
                    "indicators": {"quote": [{"close": closes}]},
                }
            ],
            "error": None,
        }
    }


def _gdelt(title: str, url: str, seen: str):
    return {
        "articles": [
            {
                "title": title,
                "url": url,
                "seendate": seen,
                "domain": "reuters.com",
            }
        ]
    }


class HistoricalAdapterTests(unittest.TestCase):
    def test_as_of_queries_hide_future_prices_and_events(self) -> None:
        store = ResearchContextStore()
        store.append_news(
            Evidence(
                ref="news:1",
                source="Synthetic",
                url="https://example.test/n1",
                published_at=DAY1,
                title="Day one",
                symbols=("AAPL",),
                knowledge_time=DAY1,
            )
        )
        store.append_news(
            Evidence(
                ref="news:2",
                source="Synthetic",
                url="https://example.test/n2",
                published_at=DAY3,
                title="Day three",
                symbols=("AAPL",),
                knowledge_time=DAY3,
            )
        )
        tape = MarketTape(
            (
                PricePrint(symbol="AAPL", price=10, knowledge_time=DAY1),
                PricePrint(symbol="AAPL", price=20, knowledge_time=DAY3),
            )
        )
        adapter = HistoricalAdapter(store, tape)
        self.assertEqual(adapter.prices_as_of(DAY1), {"AAPL": 10})
        self.assertEqual([item.ref for item in adapter.events_as_of(DAY1)], ["news:1"])
        self.assertEqual(adapter.prices_as_of(DAY3), {"AAPL": 20})
        self.assertEqual([item.ref for item in adapter.events_as_of(DAY3)], ["news:1", "news:2"])


class EquityFeedTests(unittest.TestCase):
    def test_tape_uses_bar_close_as_knowledge_time(self) -> None:
        def fetch(url: str):
            self.assertIn("AAPL", url)
            return _yahoo(((DAY1, 10.0), (DAY2, 20.0), (DAY3, 30.0)))

        tape, names = load_equity_tape(
            ("AAPL",), start=DAY1, end=DAY3, interval="15m", fetch=fetch
        )
        self.assertEqual(names["AAPL"], "Apple Inc.")
        self.assertEqual(tape.prices_as_of(DAY1), {"AAPL": 10})
        self.assertEqual(tape.next_eligible("AAPL", after=DAY1).price, 20)

    def test_rejects_crypto_pairs(self) -> None:
        with self.assertRaises(ValueError):
            load_equity_tape(("BTCUSDT",), start=DAY1, end=DAY3, fetch=lambda url: {})

    def test_news_keeps_only_in_window_sourced_articles(self) -> None:
        def fetch(url: str):
            self.assertIn("gdelt", url)
            return _gdelt(
                "Apple supplier update",
                "https://example.test/aapl",
                "20260302T000000Z",
            )

        articles = load_equity_news(("AAPL",), start=DAY1, end=DAY3, fetch=fetch)
        self.assertEqual(len(articles), 1)
        self.assertTrue(articles[0].ref.startswith("gdelt:"))
        self.assertEqual(articles[0].symbols, ("AAPL",))
        self.assertEqual(articles[0].knowledge_time, DAY2)
        self.assertEqual(articles[0].source, "reuters.com")


if __name__ == "__main__":
    unittest.main()
