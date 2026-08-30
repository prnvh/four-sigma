import unittest
from datetime import datetime, timezone

from memory import (
    InsightId,
    PortfolioRiskCalculator,
    PortfolioRiskError,
    PortfolioRiskInput,
    PortfolioSnapshot,
    Position,
    TradeCandidate,
    TradeCandidateId,
    TradeSide,
)


NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


def portfolio():
    return PortfolioSnapshot(
        cash=70_000,
        positions=(
            Position("AAA", 100, 100, 100),
            Position("BBB", -100, 100, 100),
            Position("CCC", 100, 100, 100),
        ),
        realized_pnl=0, unrealized_pnl=0, fees=0, slippage=0,
        equity=100_000, knowledge_time=NOW,
    )


def inputs(**overrides):
    values = {
        "portfolio": portfolio(),
        "sectors": {"AAA": "Tech", "BBB": "Finance", "CCC": "Tech", "DDD": "Energy"},
        "factor_loadings": {
            "AAA": {"market": 1.2, "value": -0.2},
            "BBB": {"market": 0.8, "value": 0.5},
            "CCC": {"market": 1.0, "value": 0.1},
            "DDD": {"market": 1.1, "value": 0.3},
        },
        "annualized_volatility": {"AAA": 0.20, "BBB": 0.30, "CCC": 0.25, "DDD": 0.40},
        "correlations": {
            "AAA": {"BBB": 0.2, "CCC": 0.85, "DDD": 0.1},
            "BBB": {"CCC": 0.1, "DDD": -0.2},
            "CCC": {"DDD": 0.3},
        },
        "current_drawdown": 0.08,
    }
    values.update(overrides)
    return PortfolioRiskInput(**values)


def candidate(side=TradeSide.LONG, size=0.05):
    return TradeCandidate(
        id=TradeCandidateId("trade:ddd"), instrument="DDD", direction=side,
        thesis_refs=(InsightId("insight:1"),), horizon="30 days", confidence=0.7,
        entry_conditions=("approved thesis",), exit_conditions=("invalidated",),
        proposed_size=size, knowledge_time=NOW,
    )


class PortfolioRiskTests(unittest.TestCase):
    def test_calculates_all_required_portfolio_metrics(self):
        result = PortfolioRiskCalculator().snapshot(inputs())
        self.assertAlmostEqual(result.gross_exposure, 0.30)
        self.assertAlmostEqual(result.net_exposure, 0.10)
        self.assertEqual(result.sector_exposure, {"Finance": -0.10, "Tech": 0.20})
        self.assertAlmostEqual(result.factor_exposure["market"], 0.14)
        self.assertAlmostEqual(result.factor_exposure["value"], -0.06)
        self.assertGreater(result.annualized_volatility, 0)
        self.assertAlmostEqual(result.concentration, 0.10)
        self.assertEqual(result.drawdown, 0.08)
        self.assertEqual(result.correlation_clusters, (("AAA", "CCC"),))

    def test_evaluates_before_and_after_proposed_trade(self):
        comparison = PortfolioRiskCalculator().compare(inputs(), candidate())
        self.assertAlmostEqual(comparison.before.gross_exposure, 0.30)
        self.assertAlmostEqual(comparison.after.gross_exposure, 0.35)
        self.assertAlmostEqual(comparison.after.net_exposure, 0.15)
        self.assertAlmostEqual(comparison.after.sector_exposure["Energy"], 0.05)
        self.assertEqual(comparison.trade_id, "trade:ddd")

    def test_can_evaluate_deterministically_resized_trade(self):
        calculator = PortfolioRiskCalculator()
        first = calculator.compare(inputs(), candidate(size=0.10), approved_size=0.03)
        second = calculator.compare(inputs(), candidate(size=0.10), approved_size=0.03)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.after.gross_exposure, 0.33)

    def test_short_trade_reduces_net_exposure(self):
        result = PortfolioRiskCalculator().compare(inputs(), candidate(TradeSide.SHORT))
        self.assertAlmostEqual(result.after.net_exposure, 0.05)

    def test_existing_name_is_replaced_by_target_exposure(self):
        result = PortfolioRiskCalculator().compare(
            inputs(),
            TradeCandidate(
                id=TradeCandidateId("trade:aaa-short"),
                instrument="AAA",
                direction=TradeSide.SHORT,
                thesis_refs=(InsightId("insight:1"),),
                horizon="30 days",
                confidence=0.7,
                entry_conditions=("approved thesis",),
                exit_conditions=("invalidated",),
                proposed_size=0.05,
                knowledge_time=NOW,
            ),
        )
        self.assertAlmostEqual(result.after.gross_exposure, 0.25)
        self.assertAlmostEqual(result.after.net_exposure, -0.05)

    def test_missing_price_or_metadata_fails_closed(self):
        bad_portfolio = PortfolioSnapshot(
            cash=90_000, positions=(Position("AAA", 100, 100, None),),
            realized_pnl=0, unrealized_pnl=0, fees=0, slippage=0,
            equity=100_000, knowledge_time=NOW,
        )
        with self.assertRaises(PortfolioRiskError):
            PortfolioRiskCalculator().snapshot(inputs(portfolio=bad_portfolio))
        with self.assertRaises(PortfolioRiskError):
            PortfolioRiskCalculator().snapshot(inputs(sectors={}))

    def test_missing_pairwise_correlation_fails_closed(self):
        with self.assertRaises(PortfolioRiskError):
            PortfolioRiskCalculator().snapshot(inputs(correlations={}))

    def test_invalid_approved_size_is_rejected(self):
        with self.assertRaises(PortfolioRiskError):
            PortfolioRiskCalculator().compare(inputs(), candidate(), approved_size=0.06)


if __name__ == "__main__":
    unittest.main()
