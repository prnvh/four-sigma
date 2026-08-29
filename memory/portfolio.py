from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from math import isfinite

from .types import FillId


class PortfolioError(ValueError):
    """A fill or mark cannot be applied to the book."""


class FillSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _money(value: object) -> Decimal:
    return Decimal(str(value))


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class Fill:
    """A completed paper fill. Execution policy lives elsewhere."""

    id: FillId
    instrument: str
    side: FillSide
    quantity: float
    price: float
    fee: float
    slippage: float
    knowledge_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.id, FillId):
            raise TypeError("id must be FillId")
        if not isinstance(self.side, FillSide):
            raise TypeError("side must be FillSide")
        instrument = self.instrument.strip().upper() if isinstance(self.instrument, str) else ""
        if not instrument:
            raise ValueError("instrument must be a non-empty string")
        quantity = _finite(self.quantity, "quantity")
        price = _finite(self.price, "price")
        fee = _finite(self.fee, "fee")
        slippage = _finite(self.slippage, "slippage")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if price <= 0:
            raise ValueError("price must be positive")
        if fee < 0:
            raise ValueError("fee cannot be negative")
        if slippage < 0:
            raise ValueError("slippage cannot be negative")
        _aware(self.knowledge_time, "knowledge_time")
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "slippage", slippage)


@dataclass(frozen=True, slots=True)
class Position:
    instrument: str
    quantity: float
    average_entry: float
    market_price: float | None = None
    unrealized_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    cash: float
    positions: tuple[Position, ...]
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    slippage: float
    equity: float
    knowledge_time: datetime


class PortfolioBook:
    """Deterministic cash, position, and PnL ledger. Never reads the wall clock."""

    def __init__(self, starting_cash: float, *, opened_at: datetime) -> None:
        cash = _finite(starting_cash, "starting_cash")
        if cash < 0:
            raise ValueError("starting_cash cannot be negative")
        self._opened_at = _aware(opened_at, "opened_at")
        self._cash = _money(cash)
        self._positions: dict[str, tuple[Decimal, Decimal]] = {}
        self._realized = Decimal("0")
        self._fees = Decimal("0")
        self._slippage = Decimal("0")
        self._fills: list[Fill] = []
        self._fill_ids: set[str] = set()
        self._marks: dict[str, Decimal] = {}
        self._last_time = self._opened_at

    def apply(self, fill: Fill) -> PortfolioSnapshot:
        if not isinstance(fill, Fill):
            raise TypeError("fill must be Fill")
        if fill.id.value in self._fill_ids:
            raise PortfolioError(f"duplicate fill: {fill.id.value}")
        if fill.knowledge_time < self._last_time:
            raise PortfolioError("fills must be applied in knowledge_time order")
        quantity = _money(fill.quantity)
        price = _money(fill.price)
        fee = _money(fill.fee)
        slip = _money(fill.slippage)
        notional = quantity * price
        if fill.side is FillSide.BUY:
            cash = self._cash - notional - fee - slip
            signed = quantity
        else:
            cash = self._cash + notional - fee - slip
            signed = -quantity
        if cash < 0:
            raise PortfolioError("insufficient cash")
        old_qty, old_avg = self._positions.get(fill.instrument, (Decimal("0"), Decimal("0")))
        new_qty = old_qty + signed
        realized = self._realized
        if old_qty == 0:
            updated = (signed, price)
        elif (old_qty > 0 and signed > 0) or (old_qty < 0 and signed < 0):
            updated = (new_qty, (abs(old_qty) * old_avg + quantity * price) / abs(new_qty))
        else:
            closed = min(abs(old_qty), quantity)
            if old_qty > 0:
                realized += (price - old_avg) * closed
            else:
                realized += (old_avg - price) * closed
            if new_qty == 0:
                updated = None
            elif (old_qty > 0) == (new_qty > 0):
                updated = (new_qty, old_avg)
            else:
                updated = (new_qty, price)
        self._cash = cash
        self._realized = realized
        self._fees += fee
        self._slippage += slip
        if updated is None:
            self._positions.pop(fill.instrument, None)
        else:
            self._positions[fill.instrument] = updated
        self._fills.append(fill)
        self._fill_ids.add(fill.id.value)
        self._last_time = fill.knowledge_time
        return self.snapshot()

    def mark(
        self,
        prices: Mapping[str, float],
        *,
        knowledge_time: datetime,
    ) -> PortfolioSnapshot:
        if not isinstance(prices, Mapping):
            raise TypeError("prices must be a mapping")
        when = _aware(knowledge_time, "knowledge_time")
        if when < self._last_time:
            raise PortfolioError("marks cannot precede the last book event")
        marks: dict[str, Decimal] = {}
        for raw_symbol, raw_price in prices.items():
            if not isinstance(raw_symbol, str) or not raw_symbol.strip():
                raise ValueError("mark symbol must be a non-empty string")
            price = _finite(raw_price, "mark price")
            if price <= 0:
                raise ValueError("mark price must be positive")
            marks[raw_symbol.strip().upper()] = _money(price)
        missing = sorted(symbol for symbol in self._positions if symbol not in marks)
        if missing:
            raise PortfolioError(f"missing marks for {missing}")
        self._marks = marks
        self._last_time = when
        return self.snapshot()

    def snapshot(self) -> PortfolioSnapshot:
        positions = []
        unrealized = Decimal("0")
        market_value = Decimal("0")
        for instrument in sorted(self._positions):
            quantity, average = self._positions[instrument]
            mark = self._marks.get(instrument)
            if mark is None:
                mark_price = None
                position_pnl = Decimal("0")
                market_value += quantity * average
            else:
                mark_price = float(mark)
                position_pnl = (mark - average) * quantity
                market_value += quantity * mark
            unrealized += position_pnl
            positions.append(
                Position(
                    instrument=instrument,
                    quantity=float(quantity),
                    average_entry=float(average),
                    market_price=mark_price,
                    unrealized_pnl=float(position_pnl),
                )
            )
        return PortfolioSnapshot(
            cash=float(self._cash),
            positions=tuple(positions),
            realized_pnl=float(self._realized),
            unrealized_pnl=float(unrealized),
            fees=float(self._fees),
            slippage=float(self._slippage),
            equity=float(self._cash + market_value),
            knowledge_time=self._last_time,
        )

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)
