import unittest
from datetime import datetime, timedelta, timezone

from agents import NewsAnalyst, run_backtest
from memory import (
    Direction,
    Evidence,
    MarketTape,
    PricePrint,
    PromotedInsight,
    ResearchContextStore,
    TradeCandidateStatus,
    TradeSide,
)


DAY1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
DAY2 = datetime(2026, 3, 2, tzinfo=timezone.utc)
DAY3 = datetime(2026, 3, 3, tzinfo=timezone.utc)


def article() -> Evidence:
    return Evidence(
        ref="news:1",
        source="Synthetic test source",
        url="https://example.test/news:1",
        published_at=DAY1,
        title="Synthetic test headline",
        summary="Synthetic test summary",
        symbols=("ABC",),
        knowledge_time=DAY1,
    )


def insight(
    ref: str = "insight:1",
    *,
    direction: Direction = Direction.BULLISH,
    knowledge_time: datetime = DAY1,
    valid_until: datetime | None = DAY1,
) -> PromotedInsight:
    return PromotedInsight(
        ref=ref,
        symbol="ABC",
        claim=f"Synthetic claim {ref}",
        direction=direction,
        confidence=0.8,
        evidence_refs=("news:1",),
        knowledge_time=knowledge_time,
        valid_until=valid_until,
    )


def store(*insights: PromotedInsight) -> ResearchContextStore:
    shared = ResearchContextStore()
    shared.append_news(article())
    for item in insights:
        shared.append_promoted_insight(item)
    return shared


def tape() -> MarketTape:
    return MarketTape(
        (
            PricePrint(symbol="ABC", price=10, knowledge_time=DAY1),
            PricePrint(symbol="ABC", price=20, knowledge_time=DAY2),
            PricePrint(symbol="ABC", price=22, knowledge_time=DAY3),
        )
    )


def run(**overrides: object):
    values = {
        "start": DAY1,
        "end": DAY3,
        "universe": ("ABC",),
        "agent_versions": ("trade_constructor:v1",),
        "strategy_config": {
            "starting_cash": 1000,
            "step": timedelta(days=1),
            "slippage_bps": 0,
            "fee_bps": 0,
            "max_position_pct": 0.1,
        },
        "store": store(insight()),
        "tape": tape(),
    }
    values.update(overrides)
    return run_backtest(**values)  # type: ignore[arg-type]


class BacktestRunnerTests(unittest.TestCase):
    def test_runs_clock_to_candidate_to_next_bar_fill_to_mark(self) -> None:
        result = run()
        self.assertEqual(result.invocations, ("trade_constructor:v1",))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].status, TradeCandidateStatus.APPROVED)
        self.assertEqual(result.candidates[0].direction, TradeSide.LONG)
        self.assertEqual(len(result.fills), 1)
        self.assertEqual(result.fills[0].price, 20)
        self.assertEqual(result.fills[0].knowledge_time, DAY2)
        self.assertEqual(result.final.positions[0].quantity, 0.5)
        self.assertEqual(result.final.positions[0].market_price, 22)
        self.assertEqual(result.final.unrealized_pnl, 1)
        self.assertEqual(len(result.snapshots), 3)

    def test_same_window_replays_identically(self) -> None:
        self.assertEqual(run(), run())

    def test_future_insight_is_not_visible_on_day_one(self) -> None:
        result = run(
            store=store(
                insight(
                    "insight:late",
                    knowledge_time=DAY3,
                    valid_until=DAY3,
                )
            )
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].knowledge_time, DAY3)
        self.assertEqual(result.fills, ())

    def test_no_trade_does_not_fill(self) -> None:
        result = run(store=store(insight(direction=Direction.NEUTRAL)))
        self.assertEqual(result.candidates[0].status, TradeCandidateStatus.REJECTED)
        self.assertEqual(result.fills, ())
        self.assertEqual(result.final.positions, ())

    def test_rejects_llm_agent_versions_without_a_binding(self) -> None:
        with self.assertRaises(ValueError):
            run(agent_versions=("news_analyst:v1",))

    def test_end_to_end_loads_equity_feeds_inside_the_loop(self) -> None:
        def fetch(url: str):
            if "yahoo" in url:
                return {
                    "chart": {
                        "result": [
                            {
                                "meta": {"shortName": "Apple Inc."},
                                "timestamp": [
                                    int(DAY1.timestamp()) - 900,
                                    int(DAY2.timestamp()) - 900,
                                    int(DAY3.timestamp()) - 900,
                                ],
                                "indicators": {"quote": [{"close": [10.0, 20.0, 22.0]}]},
                            }
                        ],
                        "error": None,
                    }
                }
            return {
                "articles": [
                    {
                        "title": "Apple demand rises",
                        "url": "https://example.test/aapl-1",
                        "seendate": "20260301T000000Z",
                        "domain": "reuters.com",
                    },
                    {
                        "title": "Apple guidance raised",
                        "url": "https://example.test/aapl-2",
                        "seendate": "20260301T001500Z",
                        "domain": "reuters.com",
                    },
                ]
            }

        class Model:
            def generate_json(self, *, instructions, input_data, schema):
                refs = [item["ref"] for item in input_data["articles"]]
                return {
                    "claim": "Synthetic sourced demand is rising",
                    "direction": "bullish",
                    "confidence": 0.7,
                    "horizon": "three days",
                    "evidence_refs": refs[:2],
                    "risks": ["Synthetic test risk"],
                }

        result = run_backtest(
            start=DAY1,
            end=DAY3,
            universe=("AAPL",),
            agent_versions=("news_analyst:v1", "trade_constructor:v1"),
            strategy_config={
                "starting_cash": 1000,
                "interval": "15m",
                "step": "bar",
                "max_position_pct": 0.1,
            },
            news_analyst=NewsAnalyst(Model()),
            fetch=fetch,
        )
        self.assertIn("news_analyst:v1", result.invocations)
        self.assertTrue(
            any(item.outcome.value == "APPROVED" for item in result.promotions)
        )
        self.assertEqual(result.fills[0].instrument, "AAPL")
        self.assertGreater(result.fills[0].price, 10)
        self.assertGreater(len(result.snapshots), 2)

    def test_news_finding_promotes_then_trades_next_bar(self) -> None:
        shared = ResearchContextStore()
        shared.append_news(article())
        shared.append_news(
            Evidence(
                ref="news:2",
                source="Synthetic test source",
                url="https://example.test/news:2",
                published_at=DAY1,
                title="Synthetic second headline",
                summary="Synthetic second summary",
                symbols=("ABC",),
                knowledge_time=DAY1,
            )
        )

        class Model:
            def generate_json(self, *, instructions, input_data, schema):
                return {
                    "claim": "Synthetic demand is rising",
                    "direction": "bullish",
                    "confidence": 0.7,
                    "horizon": "three days",
                    "evidence_refs": ["news:1", "news:2"],
                    "risks": ["Synthetic test risk"],
                }

        result = run(
            store=shared,
            agent_versions=("news_analyst:v1", "trade_constructor:v1"),
            news_analyst=NewsAnalyst(Model()),
        )
        self.assertIn("news_analyst:v1", result.invocations)
        self.assertEqual(result.findings[0].direction, Direction.BULLISH)
        self.assertEqual(result.promotions[0].outcome.value, "APPROVED")
        self.assertEqual(result.candidates[0].direction, TradeSide.LONG)
        self.assertEqual(result.fills[0].price, 20)


if __name__ == "__main__":
    unittest.main()
