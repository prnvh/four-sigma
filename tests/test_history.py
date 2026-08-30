import unittest
from datetime import datetime, timedelta, timezone

from agents.history_feed import (
    _seed_listed_profile,
    listed_profile,
    load_equity_news,
    load_equity_tape,
    rolling_correlations,
)
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
    def test_correlations_are_point_in_time_and_not_hard_coded_to_zero(self) -> None:
        prints = []
        for offset, left in enumerate((100, 102, 101, 105, 107, 106, 110)):
            when = DAY1 + timedelta(days=offset)
            prints.append(PricePrint("AAA", left, when))
            prints.append(PricePrint("BBB", left * 2, when))
            prints.append(PricePrint("CCC", 200 - left / 2, when))
        future = DAY1 + timedelta(days=100)
        prints.append(PricePrint("BBB", 1, future))
        matrix = rolling_correlations(
            MarketTape(tuple(prints)), ("AAA", "BBB", "CCC"), DAY1 + timedelta(days=6)
        )
        self.assertAlmostEqual(matrix["AAA"]["BBB"], 1.0)
        self.assertLess(matrix["AAA"]["CCC"], 0)
        self.assertEqual(matrix["BBB"]["AAA"], matrix["AAA"]["BBB"])

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

    def test_news_merges_parallel_symbol_chunks(self) -> None:
        def fetch(url: str):
            symbol = "MSFT" if "Microsoft" in url or "MSFT" in url else "AAPL"
            stamp = "20260302T000000Z"
            if symbol == "MSFT":
                return _gdelt("Microsoft cloud update", "https://example.test/msft", stamp)
            return _gdelt("Apple supplier update", "https://example.test/aapl", stamp)

        articles = load_equity_news(
            ("AAPL", "MSFT"),
            start=DAY1,
            end=DAY3,
            names={"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation"},
            fetch=fetch,
            max_workers=8,
        )
        self.assertEqual({item.symbols[0] for item in articles}, {"AAPL", "MSFT"})


class ListedProfileTests(unittest.TestCase):
    def test_seeds_known_listing_sector(self) -> None:
        self.assertEqual(listed_profile("aapl")[1], "Technology")
        store = ResearchContextStore()
        _seed_listed_profile(store, "AAPL", knowledge_time=DAY1)
        entity = store._company_entity_records()[0]
        self.assertEqual(entity.ticker, "AAPL")
        self.assertEqual(entity.sector, "Technology")
        self.assertEqual(entity.exchange, "NASDAQ")
        self.assertTrue(any(item.ref == "listed:profile:AAPL:sector" for item in store._company_evidence()))

    def test_unknown_symbol_is_not_invented(self) -> None:
        self.assertIsNone(listed_profile("ZZZZ"))
        store = ResearchContextStore()
        _seed_listed_profile(store, "ZZZZ", knowledge_time=DAY1)
        self.assertEqual(store._company_entity_records(), ())


if __name__ == "__main__":
    unittest.main()
