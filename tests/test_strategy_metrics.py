import unittest
from datetime import datetime, timezone

from memory import (
    Fill,
    FillId,
    FillSide,
    PortfolioSnapshot,
    Position,
    calculate_strategy_metrics,
)


START = datetime(2025, 1, 1, tzinfo=timezone.utc)
MIDDLE = datetime(2025, 7, 2, tzinfo=timezone.utc)
END = datetime(2026, 1, 1, tzinfo=timezone.utc)


def snapshot(equity, when, positions=()):
    return PortfolioSnapshot(
        cash=equity - sum(
            item.quantity * (item.market_price or item.average_entry)
            for item in positions
        ),
        positions=positions,
        realized_pnl=0,
        unrealized_pnl=0,
        fees=0,
        slippage=0,
        equity=equity,
        knowledge_time=when,
    )


def fill(identifier, side, quantity, price, when, fee=0, slippage=0):
    return Fill(
        id=FillId(identifier),
        instrument="ABC",
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        slippage=slippage,
        knowledge_time=when,
    )


class StrategyMetricsTests(unittest.TestCase):
    def test_tracks_returns_risk_costs_and_closed_trade_quality(self):
        snapshots = (
            snapshot(100, START),
            snapshot(120, MIDDLE),
            snapshot(90, END),
        )
        fills = (
            fill("fill:1", FillSide.BUY, 10, 10, START),
            fill("fill:2", FillSide.SELL, 10, 12, MIDDLE),
            fill("fill:3", FillSide.BUY, 5, 20, MIDDLE),
            fill("fill:4", FillSide.SELL, 5, 18, END, fee=1, slippage=1),
        )

        result = calculate_strategy_metrics(snapshots, fills)

        self.assertAlmostEqual(result.total_return, -0.1)
        self.assertAlmostEqual(result.cagr, -0.1, places=3)
        self.assertAlmostEqual(result.max_drawdown, 0.25)
        self.assertEqual(result.hit_rate, 0.5)
        self.assertEqual(result.profit_factor, 20 / 12)
        self.assertEqual(result.transaction_cost, 2)
        self.assertAlmostEqual(result.turnover, 410 / (310 / 3))
        self.assertIsNotNone(result.sharpe)
        self.assertIsNotNone(result.sortino)

    def test_time_weights_exposure(self):
        position = Position("ABC", 5, 10, 10)
        result = calculate_strategy_metrics(
            (
                snapshot(100, START, (position,)),
                snapshot(100, MIDDLE),
                snapshot(100, END),
            ),
            (),
        )
        expected = 0.5 * (MIDDLE - START).total_seconds() / (END - START).total_seconds()
        self.assertAlmostEqual(result.exposure, expected)

    def test_no_activity_returns_safe_undefined_ratios(self):
        result = calculate_strategy_metrics(
            (snapshot(100, START), snapshot(100, END)), ()
        )
        self.assertEqual(result.total_return, 0)
        self.assertAlmostEqual(result.cagr, 0)
        self.assertEqual(result.max_drawdown, 0)
        self.assertEqual(result.turnover, 0)
        self.assertEqual(result.transaction_cost, 0)
        self.assertEqual(result.exposure, 0)
        self.assertIsNone(result.sharpe)
        self.assertIsNone(result.sortino)
        self.assertIsNone(result.hit_rate)
        self.assertIsNone(result.profit_factor)

    def test_rejects_empty_or_out_of_order_inputs(self):
        with self.assertRaises(ValueError):
            calculate_strategy_metrics((), ())
        with self.assertRaisesRegex(ValueError, "knowledge_time order"):
            calculate_strategy_metrics(
                (snapshot(100, END), snapshot(100, START)), ()
            )


if __name__ == "__main__":
    unittest.main()
