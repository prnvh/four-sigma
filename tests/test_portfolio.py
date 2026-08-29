import unittest
from datetime import datetime, timezone

from memory import (
    Fill,
    FillId,
    FillSide,
    PortfolioBook,
    PortfolioError,
)


NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
LATER = datetime(2026, 3, 2, tzinfo=timezone.utc)


def fill(
    ref: str = "fill:1",
    *,
    instrument: str = "ABC",
    side: FillSide = FillSide.BUY,
    quantity: float = 10,
    price: float = 10,
    fee: float = 0,
    slippage: float = 0,
    knowledge_time: datetime = NOW,
) -> Fill:
    return Fill(
        id=FillId(ref),
        instrument=instrument,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        slippage=slippage,
        knowledge_time=knowledge_time,
    )


def book(starting_cash: float = 1000) -> PortfolioBook:
    return PortfolioBook(starting_cash, opened_at=NOW)


class PortfolioAccountingTests(unittest.TestCase):
    def test_buy_updates_cash_average_entry_fees_and_slippage(self) -> None:
        ledger = book()
        state = ledger.apply(fill(fee=1, slippage=0.5))
        self.assertEqual(state.cash, 898.5)
        self.assertEqual(len(state.positions), 1)
        self.assertEqual(state.positions[0].instrument, "ABC")
        self.assertEqual(state.positions[0].quantity, 10)
        self.assertEqual(state.positions[0].average_entry, 10)
        self.assertEqual(state.realized_pnl, 0)
        self.assertEqual(state.unrealized_pnl, 0)
        self.assertEqual(state.fees, 1)
        self.assertEqual(state.slippage, 0.5)
        self.assertEqual(state.equity, 998.5)

    def test_add_to_long_recomputes_average_entry(self) -> None:
        ledger = book()
        ledger.apply(fill())
        state = ledger.apply(fill("fill:2", quantity=10, price=12, knowledge_time=LATER))
        self.assertEqual(state.positions[0].quantity, 20)
        self.assertEqual(state.positions[0].average_entry, 11)
        self.assertEqual(state.cash, 780)

    def test_partial_close_realizes_pnl_and_keeps_remaining_average(self) -> None:
        ledger = book()
        ledger.apply(fill())
        state = ledger.apply(
            fill("fill:2", side=FillSide.SELL, quantity=4, price=12, knowledge_time=LATER)
        )
        self.assertEqual(state.positions[0].quantity, 6)
        self.assertEqual(state.positions[0].average_entry, 10)
        self.assertEqual(state.realized_pnl, 8)
        self.assertEqual(state.cash, 948)

    def test_mark_sets_unrealized_pnl_and_equity(self) -> None:
        ledger = book()
        ledger.apply(fill())
        state = ledger.mark({"ABC": 12}, knowledge_time=LATER)
        self.assertEqual(state.positions[0].market_price, 12)
        self.assertEqual(state.unrealized_pnl, 20)
        self.assertEqual(state.equity, 1020)

    def test_short_cover_realizes_pnl(self) -> None:
        ledger = book()
        ledger.apply(fill(side=FillSide.SELL, quantity=5, price=20))
        marked = ledger.mark({"ABC": 18}, knowledge_time=NOW)
        self.assertEqual(marked.positions[0].quantity, -5)
        self.assertEqual(marked.unrealized_pnl, 10)
        closed = ledger.apply(
            fill("fill:2", quantity=5, price=18, knowledge_time=LATER)
        )
        self.assertEqual(closed.positions, ())
        self.assertEqual(closed.realized_pnl, 10)
        self.assertEqual(closed.cash, 1010)

    def test_flip_closes_then_opens_the_other_side(self) -> None:
        ledger = book(200)
        ledger.apply(fill())
        state = ledger.apply(
            fill("fill:2", side=FillSide.SELL, quantity=15, price=12, knowledge_time=LATER)
        )
        self.assertEqual(state.positions[0].quantity, -5)
        self.assertEqual(state.positions[0].average_entry, 12)
        self.assertEqual(state.realized_pnl, 20)
        self.assertEqual(state.cash, 280)

    def test_same_fills_replay_to_the_same_snapshot(self) -> None:
        first = book()
        second = book()
        for item in (
            fill(fee=1, slippage=0.5),
            fill("fill:2", side=FillSide.SELL, quantity=4, price=12, knowledge_time=LATER),
        ):
            first.apply(item)
            second.apply(item)
        first.mark({"abc": 11}, knowledge_time=LATER)
        second.mark({"ABC": 11}, knowledge_time=LATER)
        self.assertEqual(first.snapshot(), second.snapshot())

    def test_rejects_duplicate_out_of_order_and_unfunded_fills(self) -> None:
        ledger = book(50)
        with self.assertRaises(PortfolioError):
            ledger.apply(fill())
        funded = book()
        funded.apply(fill())
        with self.assertRaises(PortfolioError):
            funded.apply(fill())
        with self.assertRaises(PortfolioError):
            funded.apply(fill("fill:2", knowledge_time=datetime(2026, 2, 1, tzinfo=timezone.utc)))

    def test_rejects_missing_marks_and_naive_times(self) -> None:
        ledger = book()
        ledger.apply(fill())
        with self.assertRaises(PortfolioError):
            ledger.mark({"XYZ": 12}, knowledge_time=LATER)
        with self.assertRaises(ValueError):
            PortfolioBook(1000, opened_at=datetime(2026, 3, 1))
        with self.assertRaises(ValueError):
            fill(knowledge_time=datetime(2026, 3, 1))


if __name__ == "__main__":
    unittest.main()
