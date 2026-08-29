import unittest
from datetime import datetime, timezone

from memory import (
    ExecutionConfig,
    ExecutionError,
    InsightId,
    MarketTape,
    PricePrint,
    SimulatedExecution,
    TradeCandidate,
    TradeCandidateId,
    TradeCandidateStatus,
    TradeSide,
)


DAY1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
DAY2 = datetime(2026, 3, 2, tzinfo=timezone.utc)
DAY3 = datetime(2026, 3, 3, tzinfo=timezone.utc)


def candidate(**overrides: object) -> TradeCandidate:
    values = {
        "id": TradeCandidateId("tc:1"),
        "instrument": "ABC",
        "direction": TradeSide.LONG,
        "thesis_refs": (InsightId("insight:1"),),
        "horizon": "test",
        "confidence": 0.6,
        "entry_conditions": ("approved insight majority",),
        "exit_conditions": ("thesis expires",),
        "proposed_size": 0.1,
        "knowledge_time": DAY1,
        "status": TradeCandidateStatus.APPROVED,
    }
    values.update(overrides)
    return TradeCandidate(**values)  # type: ignore[arg-type]


def tape() -> MarketTape:
    return MarketTape(
        (
            PricePrint(symbol="ABC", price=10, knowledge_time=DAY1),
            PricePrint(symbol="ABC", price=20, knowledge_time=DAY2),
            PricePrint(symbol="ABC", price=30, knowledge_time=DAY3),
        )
    )


class SimulatedExecutionTests(unittest.TestCase):
    def test_fills_approved_long_at_next_eligible_price(self) -> None:
        fill = SimulatedExecution(
            ExecutionConfig(slippage_bps=100, fee_bps=50)
        ).submit(candidate(), tape=tape(), equity=1000)
        self.assertIsNotNone(fill)
        assert fill is not None
        self.assertEqual(fill.side.value, "buy")
        self.assertEqual(fill.price, 20)
        self.assertEqual(fill.quantity, 5)
        self.assertEqual(fill.slippage, 1)
        self.assertEqual(fill.fee, 0.5)
        self.assertEqual(fill.knowledge_time, DAY2)

    def test_does_not_fill_on_the_decision_bar(self) -> None:
        fill = SimulatedExecution().submit(candidate(), tape=tape(), equity=1000)
        assert fill is not None
        self.assertNotEqual(fill.price, 10)
        self.assertGreater(fill.knowledge_time, DAY1)

    def test_short_uses_sell_and_same_reference_price(self) -> None:
        fill = SimulatedExecution(ExecutionConfig(slippage_bps=100)).submit(
            candidate(direction=TradeSide.SHORT), tape=tape(), equity=1000
        )
        assert fill is not None
        self.assertEqual(fill.side.value, "sell")
        self.assertEqual(fill.price, 20)
        self.assertEqual(fill.slippage, 1)

    def test_rejects_unapproved_and_skips_no_trade(self) -> None:
        broker = SimulatedExecution()
        with self.assertRaises(ExecutionError):
            broker.submit(
                candidate(status=TradeCandidateStatus.PROPOSED),
                tape=tape(),
                equity=1000,
            )
        self.assertIsNone(
            broker.submit(
                candidate(
                    direction=TradeSide.NO_TRADE,
                    proposed_size=0,
                    entry_conditions=(),
                ),
                tape=tape(),
                equity=1000,
            )
        )

    def test_missing_future_print_yields_no_fill(self) -> None:
        empty = MarketTape((PricePrint(symbol="ABC", price=10, knowledge_time=DAY1),))
        self.assertIsNone(SimulatedExecution().submit(candidate(), tape=empty, equity=1000))

    def test_prices_as_of_hides_future_prints(self) -> None:
        self.assertEqual(tape().prices_as_of(DAY1), {"ABC": 10})
        self.assertEqual(tape().prices_as_of(DAY2), {"ABC": 20})


if __name__ == "__main__":
    unittest.main()
