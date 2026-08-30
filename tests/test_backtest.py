import unittest
from datetime import datetime, timedelta, timezone

from agents import (
    CompanyAnalyst,
    NewsAnalyst,
    PortfolioRiskAgent,
    RiskAnalyst,
    TradeRiskAnalyst,
    run_backtest,
)
from memory import (
    AuditEventType,
    CompanyEntityRecord,
    CompanyRecord,
    CompanyRecordType,
    Direction,
    Evidence,
    MarketTape,
    PricePrint,
    PromotedInsight,
    ResearchContextStore,
    RiskCategory,
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
    def test_unknown_publishers_are_not_upgraded_to_reputable_newswires(self) -> None:
        from agents.backtest import _ArticleEvidence

        unknown = Evidence(
            ref="news:unknown",
            source="random-blog.invalid",
            url="https://random-blog.invalid/story",
            published_at=DAY1,
            title="Unsourced market claim",
            symbols=("ABC",),
            knowledge_time=DAY1,
        )
        self.assertEqual(_ArticleEvidence(unknown).source_class, "unverified_web")
        self.assertEqual(_ArticleEvidence(article()).source_class, "reputable_newswire")
        cnbc = Evidence(
            ref="news:cnbc",
            source="cnbc.com",
            url="https://www.cnbc.com/markets/story",
            published_at=DAY1,
            title="Market move",
            symbols=("ABC",),
            knowledge_time=DAY1,
        )
        self.assertEqual(_ArticleEvidence(cnbc).source_class, "reputable_newswire")

    def test_book_log_names_positions_and_flat_book(self) -> None:
        from memory.portfolio import PortfolioSnapshot, Position
        from agents.backtest import _held

        empty = PortfolioSnapshot(
            cash=1000,
            positions=(),
            realized_pnl=0,
            unrealized_pnl=0,
            fees=0,
            slippage=0,
            equity=1000,
            knowledge_time=DAY1,
        )
        self.assertEqual(_held(empty), "flat")
        held = PortfolioSnapshot(
            cash=500,
            positions=(
                Position("GOOGL", 7.52, 332.64, 337.52, 36.68),
                Position("AMZN", -10.75, 242.31, 241.80, 5.48),
            ),
            realized_pnl=0,
            unrealized_pnl=0,
            fees=0,
            slippage=0,
            equity=1000,
            knowledge_time=DAY1,
        )
        self.assertEqual(_held(held), "GOOGL+7.52,AMZN-10.75")

    def test_runs_clock_to_candidate_to_next_bar_fill_to_mark(self) -> None:
        result = run()
        self.assertEqual(result.invocations, ("trade_constructor:v1",))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].status, TradeCandidateStatus.CLOSED)
        self.assertEqual(result.candidates[0].direction, TradeSide.LONG)
        self.assertEqual(len(result.fills), 2)
        self.assertEqual(result.fills[0].price, 20)
        self.assertEqual(result.fills[0].knowledge_time, DAY2)
        self.assertEqual(result.fills[1].price, 22)
        self.assertEqual(result.final.positions, ())
        self.assertEqual(result.final.realized_pnl, 10)
        self.assertEqual(len(result.snapshots), 3)
        self.assertAlmostEqual(result.metrics.total_return, 0.01)
        self.assertGreater(result.metrics.turnover, 0)

        transitions = [
            event.details.get("to_status")
            for event in result.audit_events
            if event.event_type is AuditEventType.TRADE_STATUS_CHANGED
        ]
        self.assertEqual(
            transitions,
            ["risk_reviewed", "approved", "submitted", "filled", "closed"],
        )

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

    def test_positive_risk_reduction_is_executed_instead_of_rejected(self) -> None:
        class ReduceModel:
            def generate_json(self, *, instructions, input_data, schema):
                return {
                    "action": "reduce",
                    "size": 0.05,
                    "rationale": "Use a smaller target while retaining the thesis.",
                    "evidence_refs": ["insight:1"],
                }

        result = run(
            agent_versions=("trade_constructor:v1", "risk_llm:trade_v1"),
            trade_risk=TradeRiskAnalyst(ReduceModel()),
        )
        self.assertEqual(result.candidates[0].status, TradeCandidateStatus.CLOSED)
        self.assertAlmostEqual(result.fills[0].quantity, 2.5)
        self.assertEqual(result.fills[0].price, 20)

    def test_reduce_does_not_cut_a_working_long(self) -> None:
        from memory import FillSide

        class ReduceModel:
            def generate_json(self, *, instructions, input_data, schema):
                return {
                    "action": "reduce",
                    "size": 0.05,
                    "rationale": "Keep the thesis but cut the winner.",
                    "evidence_refs": ["insight:1"],
                }

        result = run(
            store=store(insight(valid_until=DAY3)),
            agent_versions=("trade_constructor:v1", "risk_llm:trade_v1"),
            trade_risk=TradeRiskAnalyst(ReduceModel()),
        )
        buys = [item for item in result.fills if item.side is FillSide.BUY]
        sells = [item for item in result.fills if item.side is FillSide.SELL]
        self.assertEqual(len(buys), 1)
        self.assertEqual(sells, [])
        self.assertTrue(any(item.quantity > 0 for item in result.final.positions))

    def test_does_not_flip_a_working_long(self) -> None:
        from memory import FillSide

        result = run(
            store=store(
                insight(valid_until=DAY3),
                insight(
                    "insight:bear",
                    direction=Direction.BEARISH,
                    knowledge_time=DAY3,
                    valid_until=DAY3,
                ),
            )
        )
        sells = [item for item in result.fills if item.side is FillSide.SELL]
        self.assertEqual(sells, [])
        self.assertTrue(any(item.quantity > 0 for item in result.final.positions))
        self.assertFalse(any(item.quantity < 0 for item in result.final.positions))

    def test_rejects_llm_agent_versions_without_a_binding(self) -> None:
        with self.assertRaises(ValueError):
            run(agent_versions=("news_analyst:v1",))
        with self.assertRaises(ValueError):
            run(agent_versions=("company_analyst:v1", "trade_constructor:v1"))
        with self.assertRaises(ValueError):
            run(agent_versions=("risk_llm:v1", "trade_constructor:v1"))
        with self.assertRaises(ValueError):
            run(agent_versions=("portfolio_risk:v1", "trade_constructor:v1"))

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
                        "title": "Apple supplier orders rise",
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

    def test_news_waits_three_hours_between_calls(self) -> None:
        shared = ResearchContextStore()
        prints = []
        for hour in range(5):
            when = DAY1 + timedelta(hours=hour)
            shared.append_news(
                Evidence(
                    ref=f"news:{hour}:a",
                    source="Synthetic test source",
                    url=f"https://example.test/{hour}-a",
                    published_at=when,
                    title=f"Synthetic headline {hour}a",
                    summary=f"Synthetic summary {hour}a",
                    symbols=("ABC",),
                    knowledge_time=when,
                )
            )
            shared.append_news(
                Evidence(
                    ref=f"news:{hour}:b",
                    source="Synthetic test source",
                    url=f"https://example.test/{hour}-b",
                    published_at=when,
                    title=f"Synthetic headline {hour}b",
                    summary=f"Synthetic summary {hour}b",
                    symbols=("ABC",),
                    knowledge_time=when,
                )
            )
            prints.append(PricePrint(symbol="ABC", price=10 + hour, knowledge_time=when))

        class Model:
            def __init__(self) -> None:
                self.calls = 0

            def generate_json(self, *, instructions, input_data, schema):
                self.calls += 1
                refs = [item["ref"] for item in input_data["articles"]]
                return {
                    "claim": "Synthetic demand is rising",
                    "direction": "bullish",
                    "confidence": 0.7,
                    "horizon": "three days",
                    "evidence_refs": refs[:2],
                    "risks": ["Synthetic test risk"],
                }

        model = Model()
        result = run(
            start=DAY1,
            end=DAY1 + timedelta(hours=4),
            store=shared,
            tape=MarketTape(tuple(prints)),
            strategy_config={
                "starting_cash": 1000,
                "step": "bar",
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "news_cadence": timedelta(hours=3),
            },
            agent_versions=("news_analyst:v1", "trade_constructor:v1"),
            news_analyst=NewsAnalyst(model),
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(result.invocations.count("news_analyst:v1"), 2)

    def test_bound_llm_agents_all_run(self) -> None:
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
        shared.append_company_record(
            CompanyRecord(
                ref="yahoo:profile:ABC:sector",
                symbol="ABC",
                source="Yahoo Finance",
                url="https://example.test/profile",
                knowledge_time=DAY1,
                record_type=CompanyRecordType.COMPANY_PROFILE,
                label="sector",
                value="Technology",
            )
        )
        shared.append_company_record(
            CompanyRecord(
                ref="yahoo:profile:ABC:industry",
                symbol="ABC",
                source="Yahoo Finance",
                url="https://example.test/profile",
                knowledge_time=DAY1,
                record_type=CompanyRecordType.COMPANY_PROFILE,
                label="industry",
                value="Consumer Electronics",
            )
        )
        shared.append_company_entity(
            CompanyEntityRecord(
                ticker="ABC",
                exchange="TEST",
                sector="Technology",
                industry="Consumer Electronics",
                identifiers={"yahoo": "ABC"},
                fundamental_references=(
                    "yahoo:profile:ABC:sector",
                    "yahoo:profile:ABC:industry",
                ),
                knowledge_time=DAY1,
            )
        )
        prints = MarketTape(
            (
                PricePrint(
                    symbol="ABC",
                    price=10,
                    knowledge_time=DAY1 - timedelta(days=10),
                    volume=1_000_000,
                ),
                PricePrint(
                    symbol="ABC",
                    price=10.2,
                    knowledge_time=DAY1 - timedelta(days=5),
                    volume=1_000_000,
                ),
                PricePrint(
                    symbol="ABC",
                    price=10.1,
                    knowledge_time=DAY1 - timedelta(days=1),
                    volume=1_000_000,
                ),
                PricePrint(symbol="ABC", price=10, knowledge_time=DAY1, volume=1_000_000),
                PricePrint(symbol="ABC", price=10.12, knowledge_time=DAY2, volume=1_000_000),
                PricePrint(symbol="ABC", price=10.18, knowledge_time=DAY3, volume=1_000_000),
            )
        )

        class Model:
            def generate_json(self, *, instructions, input_data, schema):
                required = set(schema.get("required", []))
                if "claim" in required:
                    return {
                        "claim": "Synthetic demand is rising",
                        "direction": "bullish",
                        "confidence": 0.7,
                        "horizon": "three days",
                        "evidence_refs": ["news:1", "news:2"],
                        "risks": ["Synthetic test risk"],
                    }
                if "company_thesis" in required:
                    refs = [item["ref"] for item in input_data.get("approved_insights", [])]
                    refs.extend(item["ref"] for item in input_data.get("recent_events", []))
                    refs.extend(item["ref"] for item in input_data.get("company_facts", []))
                    return {
                        "company_thesis": "Synthetic sourced company thesis",
                        "bull_case": "Synthetic bull case",
                        "bear_case": "Synthetic bear case",
                        "catalysts": ["Synthetic catalyst"],
                        "risks": ["Synthetic risk"],
                        "confidence": 0.6,
                        "time_horizon": "one month",
                        "evidence_refs": refs[:1],
                        "supports": refs[:1] if refs and refs[0].startswith("insight:") else [],
                        "contradicts": [],
                        "supersedes": [],
                    }
                if "overall_risk_score" in required:
                    refs = [
                        item["ref"]
                        for key in (
                            "company_analyses",
                            "company_records",
                            "promoted_insights",
                            "market_features",
                        )
                        for item in input_data.get(key, [])
                    ]
                    assessed = {RiskCategory.MARKET.value}
                    return {
                        "overall_risk_score": 40,
                        "success_probability_pct": 40,
                        "neutral_probability_pct": 35,
                        "failure_probability_pct": 25,
                        "risk_factors": [
                            {
                                "category": "market",
                                "probability_pct": 30,
                                "severity": 3,
                                "impact": "Synthetic market impact",
                                "evidence_refs": refs[:1],
                                "mitigants": ["Synthetic mitigant"],
                            }
                        ],
                        "hidden_assumptions": ["Synthetic assumption"],
                        "second_order_effects": ["Synthetic second order"],
                        "success_conditions": ["Synthetic success"],
                        "failure_conditions": ["Synthetic failure"],
                        "coverage_gaps": [
                            item.value for item in RiskCategory if item.value not in assessed
                        ],
                        "evidence_refs": refs[:1],
                    }
                if "recommendation" in required:
                    refs = [
                        item["ref"]
                        for item in input_data.get("selected_insight_summaries", [])
                    ]
                    size = input_data["proposed_trade"]["proposed_size"]
                    return {
                        "recommendation": "approve",
                        "recommended_size": size,
                        "rationale": "Synthetic portfolio approval",
                        "risk_flags": ["Synthetic flag"],
                        "insight_refs": refs,
                    }
                refs = [item["ref"] for item in input_data.get("insights", [])]
                refs.extend(item["ref"] for item in input_data.get("articles", []))
                return {
                    "action": "allow",
                    "size": input_data["proposed_trade"]["proposed_size"],
                    "rationale": "Synthetic trade-risk allow",
                    "evidence_refs": refs[:1],
                }

        model = Model()
        result = run(
            store=shared,
            tape=prints,
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "max_annualized_volatility": 10.0,
            },
            agent_versions=(
                "news_analyst:v1",
                "company_analyst:v1",
                "risk_llm:v1",
                "trade_constructor:v1",
                "portfolio_risk:v1",
                "risk_llm:trade_v1",
            ),
            news_analyst=NewsAnalyst(model),
            company_analyst=CompanyAnalyst(model),
            risk_analyst=RiskAnalyst(model),
            portfolio_risk=PortfolioRiskAgent(model),
            trade_risk=TradeRiskAnalyst(model),
        )
        self.assertEqual(result.invocations.count("news_analyst:v1"), 1)
        self.assertEqual(result.invocations.count("company_analyst:v1"), 1)
        self.assertEqual(result.invocations.count("risk_llm:v1"), 1)
        self.assertEqual(result.invocations.count("portfolio_risk:v1"), 1)
        self.assertEqual(result.invocations.count("risk_llm:trade_v1"), 1)
        self.assertEqual(result.company_analyses[0].company_thesis, "Synthetic sourced company thesis")
        self.assertEqual(result.risk_analyses[0].overall_risk_score, 40)
        self.assertEqual(result.candidates[0].direction, TradeSide.LONG)

    def test_multiple_symbols_run_agent_tracks_in_parallel(self) -> None:
        import threading
        import time

        shared = ResearchContextStore()
        prints = []
        for symbol in ("ABC", "XYZ"):
            shared.append_news(
                Evidence(
                    ref=f"news:{symbol}:1",
                    source="Synthetic test source",
                    url=f"https://example.test/{symbol}-1",
                    published_at=DAY1,
                    title=f"Synthetic {symbol} headline",
                    summary=f"Synthetic {symbol} summary",
                    symbols=(symbol,),
                    knowledge_time=DAY1,
                )
            )
            shared.append_news(
                Evidence(
                    ref=f"news:{symbol}:2",
                    source="Synthetic test source",
                    url=f"https://example.test/{symbol}-2",
                    published_at=DAY1,
                    title=f"Synthetic {symbol} second headline",
                    summary=f"Synthetic {symbol} second summary",
                    symbols=(symbol,),
                    knowledge_time=DAY1,
                )
            )
            shared.append_company_record(
                CompanyRecord(
                    ref=f"yahoo:profile:{symbol}:sector",
                    symbol=symbol,
                    source="Yahoo Finance",
                    url="https://example.test/profile",
                    knowledge_time=DAY1,
                    record_type=CompanyRecordType.COMPANY_PROFILE,
                    label="sector",
                    value="Technology",
                )
            )
            shared.append_company_record(
                CompanyRecord(
                    ref=f"yahoo:profile:{symbol}:industry",
                    symbol=symbol,
                    source="Yahoo Finance",
                    url="https://example.test/profile",
                    knowledge_time=DAY1,
                    record_type=CompanyRecordType.COMPANY_PROFILE,
                    label="industry",
                    value="Software",
                )
            )
            shared.append_company_entity(
                CompanyEntityRecord(
                    ticker=symbol,
                    exchange="TEST",
                    sector="Technology",
                    industry="Software",
                    identifiers={"yahoo": symbol},
                    fundamental_references=(
                        f"yahoo:profile:{symbol}:sector",
                        f"yahoo:profile:{symbol}:industry",
                    ),
                    knowledge_time=DAY1,
                )
            )
            prints.extend(
                (
                    PricePrint(
                        symbol=symbol,
                        price=10,
                        knowledge_time=DAY1 - timedelta(days=10),
                        volume=1_000_000,
                    ),
                    PricePrint(
                        symbol=symbol,
                        price=10.2,
                        knowledge_time=DAY1 - timedelta(days=5),
                        volume=1_000_000,
                    ),
                    PricePrint(
                        symbol=symbol,
                        price=10.1,
                        knowledge_time=DAY1 - timedelta(days=1),
                        volume=1_000_000,
                    ),
                    PricePrint(symbol=symbol, price=10, knowledge_time=DAY1, volume=1_000_000),
                    PricePrint(symbol=symbol, price=10.12, knowledge_time=DAY2, volume=1_000_000),
                    PricePrint(symbol=symbol, price=10.18, knowledge_time=DAY3, volume=1_000_000),
                )
            )

        class ConcurrentModel:
            def __init__(self) -> None:
                self._lock = threading.Lock()
                self._active = 0
                self.max_active = 0

            def generate_json(self, *, instructions, input_data, schema):
                with self._lock:
                    self._active += 1
                    self.max_active = max(self.max_active, self._active)
                time.sleep(0.05)
                try:
                    required = set(schema.get("required", []))
                    symbol = input_data.get("symbol") or input_data.get(
                        "proposed_trade", {}
                    ).get("instrument")
                    if "claim" in required:
                        refs = [item["ref"] for item in input_data["articles"]]
                        return {
                            "claim": "Synthetic demand is rising",
                            "direction": "bullish",
                            "confidence": 0.7,
                            "horizon": "three days",
                            "evidence_refs": refs[:2],
                            "risks": ["Synthetic test risk"],
                        }
                    if "company_thesis" in required:
                        refs = [item["ref"] for item in input_data.get("approved_insights", [])]
                        refs.extend(item["ref"] for item in input_data.get("recent_events", []))
                        refs.extend(item["ref"] for item in input_data.get("company_facts", []))
                        return {
                            "company_thesis": f"Synthetic sourced company thesis {symbol}",
                            "bull_case": "Synthetic bull case",
                            "bear_case": "Synthetic bear case",
                            "catalysts": ["Synthetic catalyst"],
                            "risks": ["Synthetic risk"],
                            "confidence": 0.6,
                            "time_horizon": "one month",
                            "evidence_refs": refs[:1],
                            "supports": refs[:1] if refs and refs[0].startswith("insight:") else [],
                            "contradicts": [],
                            "supersedes": [],
                        }
                    if "overall_risk_score" in required:
                        refs = [
                            item["ref"]
                            for key in (
                                "company_analyses",
                                "company_records",
                                "promoted_insights",
                                "market_features",
                            )
                            for item in input_data.get(key, [])
                        ]
                        assessed = {RiskCategory.MARKET.value}
                        return {
                            "overall_risk_score": 40,
                            "success_probability_pct": 40,
                            "neutral_probability_pct": 35,
                            "failure_probability_pct": 25,
                            "risk_factors": [
                                {
                                    "category": "market",
                                    "probability_pct": 30,
                                    "severity": 3,
                                    "impact": "Synthetic market impact",
                                    "evidence_refs": refs[:1],
                                    "mitigants": ["Synthetic mitigant"],
                                }
                            ],
                            "hidden_assumptions": ["Synthetic assumption"],
                            "second_order_effects": ["Synthetic second order"],
                            "success_conditions": ["Synthetic success"],
                            "failure_conditions": ["Synthetic failure"],
                            "coverage_gaps": [
                                item.value
                                for item in RiskCategory
                                if item.value not in assessed
                            ],
                            "evidence_refs": refs[:1],
                        }
                    if "recommendation" in required:
                        refs = [
                            item["ref"]
                            for item in input_data.get("selected_insight_summaries", [])
                        ]
                        size = input_data["proposed_trade"]["proposed_size"]
                        return {
                            "recommendation": "approve",
                            "recommended_size": size,
                            "rationale": "Synthetic portfolio approval",
                            "risk_flags": ["Synthetic flag"],
                            "insight_refs": refs,
                        }
                    refs = [item["ref"] for item in input_data.get("insights", [])]
                    refs.extend(item["ref"] for item in input_data.get("articles", []))
                    return {
                        "action": "allow",
                        "size": input_data["proposed_trade"]["proposed_size"],
                        "rationale": "Synthetic trade-risk allow",
                        "evidence_refs": refs[:1],
                    }
                finally:
                    with self._lock:
                        self._active -= 1

        model = ConcurrentModel()
        result = run(
            store=shared,
            tape=MarketTape(tuple(prints)),
            universe=("ABC", "XYZ"),
            strategy_config={
                "starting_cash": 10_000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "max_annualized_volatility": 10.0,
            },
            agent_versions=(
                "news_analyst:v1",
                "company_analyst:v1",
                "risk_llm:v1",
                "trade_constructor:v1",
                "portfolio_risk:v1",
                "risk_llm:trade_v1",
            ),
            news_analyst=NewsAnalyst(model),
            company_analyst=CompanyAnalyst(model),
            risk_analyst=RiskAnalyst(model),
            portfolio_risk=PortfolioRiskAgent(model),
            trade_risk=TradeRiskAnalyst(model),
        )
        self.assertGreaterEqual(model.max_active, 2)
        self.assertEqual({item.subject for item in result.findings}, {"ABC", "XYZ"})
        self.assertEqual({item.symbol for item in result.company_analyses}, {"ABC", "XYZ"})
        self.assertEqual({item.instrument for item in result.candidates}, {"ABC", "XYZ"})

    def test_simultaneous_names_share_the_book(self) -> None:
        from memory import FillSide

        shared = ResearchContextStore()
        prints = []
        for symbol in ("AAA", "BBB"):
            shared.append_news(
                Evidence(
                    ref=f"news:{symbol}",
                    source="Synthetic test source",
                    url=f"https://example.test/{symbol}",
                    published_at=DAY1,
                    title=f"Synthetic {symbol} headline",
                    summary=f"Synthetic {symbol} summary",
                    symbols=(symbol,),
                    knowledge_time=DAY1,
                )
            )
            shared.append_promoted_insight(
                PromotedInsight(
                    ref=f"insight:{symbol}",
                    symbol=symbol,
                    claim=f"Synthetic claim {symbol}",
                    direction=Direction.BULLISH,
                    confidence=0.8,
                    evidence_refs=(f"news:{symbol}",),
                    knowledge_time=DAY1,
                    valid_until=DAY3,
                )
            )
            prints.extend(
                (
                    PricePrint(symbol=symbol, price=10, knowledge_time=DAY1),
                    PricePrint(symbol=symbol, price=10, knowledge_time=DAY2),
                    PricePrint(symbol=symbol, price=10, knowledge_time=DAY3),
                )
            )
        result = run(
            store=shared,
            tape=MarketTape(tuple(prints)),
            universe=("AAA", "BBB"),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 1.0,
            },
        )
        buys = [item for item in result.fills if item.side is FillSide.BUY]
        self.assertEqual({item.instrument for item in buys}, {"AAA", "BBB"})
        for item in buys:
            self.assertAlmostEqual(item.quantity * item.price, 500, places=4)

    def test_stop_flattens_a_losing_long_and_waits_to_reenter(self) -> None:
        from memory import FillSide

        day4 = DAY3 + timedelta(days=1)
        result = run(
            end=day4,
            store=store(
                insight(valid_until=day4),
                insight(
                    "insight:2",
                    knowledge_time=DAY3,
                    valid_until=day4,
                ),
            ),
            tape=MarketTape(
                (
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY1),
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY2),
                    PricePrint(symbol="ABC", price=90, knowledge_time=DAY3),
                    PricePrint(symbol="ABC", price=90, knowledge_time=day4),
                )
            ),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "stop_loss_pct": 0.05,
                "stop_reentry_cooldown": timedelta(days=3),
            },
        )
        buys = [item for item in result.fills if item.side is FillSide.BUY]
        sells = [item for item in result.fills if item.side is FillSide.SELL]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(sells), 1)
        self.assertEqual(buys[0].price, 100)
        self.assertEqual(sells[0].price, 90)
        self.assertEqual(result.final.positions, ())
        self.assertLess(result.final.realized_pnl, 0)
        self.assertTrue(
            any(
                item.status is TradeCandidateStatus.REJECTED
                and item.direction is TradeSide.LONG
                for item in result.candidates
            )
        )

    def test_stop_allows_same_side_reentry_after_new_evidence(self) -> None:
        from memory import FillSide

        day4 = DAY3 + timedelta(days=1)
        day5 = DAY3 + timedelta(days=2)
        result = run(
            end=day5,
            store=store(
                insight(valid_until=day5),
                insight(
                    "insight:fresh",
                    knowledge_time=day4,
                    valid_until=day5,
                ),
            ),
            tape=MarketTape(
                (
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY1),
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY2),
                    PricePrint(symbol="ABC", price=90, knowledge_time=DAY3),
                    PricePrint(symbol="ABC", price=102, knowledge_time=day4),
                    PricePrint(symbol="ABC", price=103, knowledge_time=day5),
                )
            ),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "stop_loss_pct": 0.05,
                "stop_reentry_cooldown": timedelta(days=1),
            },
        )
        buys = [item for item in result.fills if item.side is FillSide.BUY]
        self.assertGreaterEqual(len(buys), 2)
        self.assertEqual(buys[0].price, 100)
        self.assertEqual(buys[-1].price, 103)

    def test_trailing_stop_protects_a_winner_after_retracement(self) -> None:
        from memory import FillSide

        day4 = DAY3 + timedelta(days=1)
        day5 = DAY3 + timedelta(days=2)
        result = run(
            end=day5,
            store=store(insight(valid_until=day5)),
            tape=MarketTape(
                (
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY1),
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY2),
                    PricePrint(symbol="ABC", price=106, knowledge_time=DAY3),
                    PricePrint(symbol="ABC", price=103, knowledge_time=day4),
                    PricePrint(symbol="ABC", price=103, knowledge_time=day5),
                )
            ),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "stop_loss_pct": 0.5,
                "trailing_stop_activation_pct": 0.04,
                "trailing_stop_floor_pct": 0.02,
                "trailing_stop_volatility_multiple": 0,
                "stop_reentry_cooldown": timedelta(days=3),
            },
        )
        buys = [item for item in result.fills if item.side is FillSide.BUY]
        sells = [item for item in result.fills if item.side is FillSide.SELL]
        self.assertEqual(len(buys), 1)
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0].price, 103)
        self.assertGreater(result.final.realized_pnl, 0)
        self.assertEqual(result.final.positions, ())

    def test_stop_then_latest_bearish_insight_can_short(self) -> None:
        from memory import FillSide

        day4 = DAY3 + timedelta(days=1)
        day5 = DAY3 + timedelta(days=2)
        result = run(
            end=day5,
            store=store(
                insight(valid_until=day5),
                insight(
                    "insight:bear",
                    direction=Direction.BEARISH,
                    knowledge_time=DAY3,
                    valid_until=day5,
                ),
            ),
            tape=MarketTape(
                (
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY1),
                    PricePrint(symbol="ABC", price=100, knowledge_time=DAY2),
                    PricePrint(symbol="ABC", price=90, knowledge_time=DAY3),
                    PricePrint(symbol="ABC", price=90, knowledge_time=day4),
                    PricePrint(symbol="ABC", price=88, knowledge_time=day5),
                )
            ),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
                "stop_loss_pct": 0.05,
                "stop_reentry_cooldown": timedelta(days=1),
            },
        )
        self.assertTrue(
            any(item.direction is TradeSide.SHORT for item in result.candidates)
        )
        sells = [item for item in result.fills if item.side is FillSide.SELL]
        self.assertGreaterEqual(len(sells), 2)
        self.assertEqual(sells[0].price, 90)
        self.assertEqual(sells[-1].price, 88)
        self.assertTrue(any(item.quantity < 0 for item in result.final.positions))

    def test_position_limits_keep_single_name_at_the_configured_cap(self) -> None:
        from agents.backtest import _position_limits

        full = _position_limits(1.0)
        self.assertEqual(full.max_single_name_concentration, 1.0)
        self.assertEqual(full.max_net_exposure, 1.0)
        sleeve = _position_limits(0.25)
        self.assertEqual(sleeve.max_position_pct, 0.25)
        self.assertEqual(sleeve.max_single_name_concentration, 0.25)
        self.assertEqual(sleeve.max_net_exposure, 1.0)
        self.assertEqual(sleeve.max_gross_exposure, 1.2)

    def test_new_entry_defers_when_failure_exceeds_success(self) -> None:
        from agents.backtest import _defer_weak_thesis
        from types import SimpleNamespace

        weak = SimpleNamespace(failure_probability_pct=55, success_probability_pct=30)
        close = SimpleNamespace(failure_probability_pct=45, success_probability_pct=35)
        strong = SimpleNamespace(failure_probability_pct=25, success_probability_pct=40)
        self.assertTrue(_defer_weak_thesis(weak, 0))
        self.assertFalse(_defer_weak_thesis(weak, 5))
        self.assertFalse(_defer_weak_thesis(close, 0))
        self.assertFalse(_defer_weak_thesis(strong, 0))
        self.assertFalse(_defer_weak_thesis(None, 0))

    def test_visible_sector_falls_back_to_listed_profile(self) -> None:
        from agents.backtest import _visible_sector

        empty = ResearchContextStore()
        self.assertEqual(_visible_sector(empty, "AAPL", DAY1), "Technology")
        self.assertEqual(_visible_sector(empty, "AMZN", DAY1), "Consumer Cyclical")
        self.assertIsNone(_visible_sector(empty, "ZZZZ", DAY1))


if __name__ == "__main__":
    unittest.main()
