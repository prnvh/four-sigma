from __future__ import annotations

from collections.abc import Sequence
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from math import isfinite

from .portfolio import Fill, FillSide
from .types import FillId, TradeCandidate, TradeCandidateStatus, TradeSide


class ExecutionError(ValueError):
    """An order cannot be turned into a paper fill."""


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _bps(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} cannot be negative")
    return number


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    slippage_bps: float = 0.0
    fee_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "slippage_bps", _bps(self.slippage_bps, "slippage_bps"))
        object.__setattr__(self, "fee_bps", _bps(self.fee_bps, "fee_bps"))


@dataclass(frozen=True, slots=True)
class PricePrint:
    symbol: str
    price: float
    knowledge_time: datetime
    volume: float | None = None

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper() if isinstance(self.symbol, str) else ""
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        price = _finite(self.price, "price")
        if price <= 0:
            raise ValueError("price must be positive")
        _aware(self.knowledge_time, "knowledge_time")
        volume = self.volume
        if volume is not None:
            volume = _finite(volume, "volume")
            if volume < 0:
                raise ValueError("volume cannot be negative")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "volume", volume)


class MarketTape:
    """Append-only prints. Queries never see knowledge_time after the as-of clock."""

    def __init__(self, prints: Sequence[PricePrint] = ()) -> None:
        if any(not isinstance(item, PricePrint) for item in prints):
            raise TypeError("prints must contain only PricePrint values")
        self._prints = sorted(prints, key=lambda item: (item.knowledge_time, item.symbol))
        self._by_symbol: dict[str, list[PricePrint]] = {}
        self._times_by_symbol: dict[str, list[datetime]] = {}
        for item in self._prints:
            self._by_symbol.setdefault(item.symbol, []).append(item)
            self._times_by_symbol.setdefault(item.symbol, []).append(item.knowledge_time)

    def add(self, print_: PricePrint) -> None:
        if not isinstance(print_, PricePrint):
            raise TypeError("print must be PricePrint")
        global_keys = [(item.knowledge_time, item.symbol) for item in self._prints]
        global_index = bisect_right(global_keys, (print_.knowledge_time, print_.symbol))
        self._prints.insert(global_index, print_)
        times = self._times_by_symbol.setdefault(print_.symbol, [])
        symbol_prints = self._by_symbol.setdefault(print_.symbol, [])
        symbol_index = bisect_right(times, print_.knowledge_time)
        times.insert(symbol_index, print_.knowledge_time)
        symbol_prints.insert(symbol_index, print_)

    def next_eligible(self, symbol: str, *, after: datetime) -> PricePrint | None:
        symbol = symbol.strip().upper()
        after = _aware(after, "after")
        times = self._times_by_symbol.get(symbol, ())
        index = bisect_right(times, after)
        found = self._by_symbol.get(symbol, ())
        return found[index] if index < len(found) else None

    def prices_as_of(self, when: datetime) -> dict[str, float]:
        when = _aware(when, "when")
        latest: dict[str, float] = {}
        for symbol, times in self._times_by_symbol.items():
            index = bisect_right(times, when) - 1
            if index >= 0:
                latest[symbol] = self._by_symbol[symbol][index].price
        return latest

    def prints_for(
        self,
        symbol: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[PricePrint, ...]:
        symbol = symbol.strip().upper()
        times = self._times_by_symbol.get(symbol, ())
        left = 0 if start is None else bisect_left(times, _aware(start, "start"))
        right = len(times) if end is None else bisect_right(times, _aware(end, "end"))
        return tuple(self._by_symbol.get(symbol, ())[left:right])

    @property
    def prints(self) -> tuple[PricePrint, ...]:
        return tuple(self._prints)


class SimulatedExecution:
    """Market-on-next-print fills. Slippage and fees are cash costs on the reference price."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def submit(
        self,
        candidate: TradeCandidate,
        *,
        tape: MarketTape,
        equity: float,
        existing_quantity: float = 0,
        cash: float | None = None,
    ) -> Fill | None:
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if not isinstance(tape, MarketTape):
            raise TypeError("tape must be MarketTape")
        if candidate.status is not TradeCandidateStatus.APPROVED:
            raise ExecutionError("only approved candidates can be submitted")
        equity_value = _finite(equity, "equity")
        if equity_value <= 0:
            raise ExecutionError("equity must be positive")
        cash_value = equity_value if cash is None else _finite(cash, "cash")
        if cash_value < 0:
            raise ExecutionError("cash cannot be negative")
        current = Decimal(str(_finite(existing_quantity, "existing_quantity")))
        nxt = tape.next_eligible(candidate.instrument, after=candidate.knowledge_time)
        if nxt is None:
            return None
        reference = Decimal(str(nxt.price))
        if candidate.direction is TradeSide.NO_TRADE:
            target = Decimal("0")
        else:
            target = Decimal(str(equity_value)) * Decimal(str(candidate.proposed_size)) / reference
            if candidate.direction is TradeSide.SHORT:
                target = -target
        delta = target - current
        cost_rate = (
            Decimal(str(self.config.slippage_bps)) + Decimal(str(self.config.fee_bps))
        ) / Decimal("10000")
        if delta > 0:
            affordable = (Decimal(str(cash_value)) / (reference * (Decimal("1") + cost_rate))).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            if affordable <= 0:
                return None
            if delta > affordable:
                delta = affordable
        if delta == 0:
            return None
        quantity = abs(delta)
        slip = quantity * reference * Decimal(str(self.config.slippage_bps)) / Decimal("10000")
        fee = quantity * reference * Decimal(str(self.config.fee_bps)) / Decimal("10000")
        side = FillSide.BUY if delta > 0 else FillSide.SELL
        return Fill(
            id=FillId(f"{candidate.id.value}:{nxt.knowledge_time.isoformat()}"),
            instrument=candidate.instrument,
            side=side,
            quantity=float(quantity),
            price=float(reference),
            fee=float(fee),
            slippage=float(slip),
            knowledge_time=nxt.knowledge_time,
        )

    def flatten(
        self,
        instrument: str,
        existing_quantity: float,
        *,
        tape: MarketTape,
        after: datetime,
    ) -> Fill | None:
        current = _finite(existing_quantity, "existing_quantity")
        if current == 0:
            return None
        nxt = tape.next_eligible(instrument, after=after)
        if nxt is None:
            return None
        quantity = abs(current)
        reference = Decimal(str(nxt.price))
        slip = (
            Decimal(str(quantity))
            * reference
            * Decimal(str(self.config.slippage_bps))
            / Decimal("10000")
        )
        fee = (
            Decimal(str(quantity))
            * reference
            * Decimal(str(self.config.fee_bps))
            / Decimal("10000")
        )
        return Fill(
            id=FillId(f"flat:{instrument}:{nxt.knowledge_time.isoformat()}"),
            instrument=instrument,
            side=FillSide.SELL if current > 0 else FillSide.BUY,
            quantity=quantity,
            price=float(reference),
            fee=float(fee),
            slippage=float(slip),
            knowledge_time=nxt.knowledge_time,
        )
