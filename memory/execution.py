from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
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

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper() if isinstance(self.symbol, str) else ""
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        price = _finite(self.price, "price")
        if price <= 0:
            raise ValueError("price must be positive")
        _aware(self.knowledge_time, "knowledge_time")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "price", price)


class MarketTape:
    """Append-only prints. Queries never see knowledge_time after the as-of clock."""

    def __init__(self, prints: Sequence[PricePrint] = ()) -> None:
        self._prints: list[PricePrint] = []
        for item in prints:
            self.add(item)

    def add(self, print_: PricePrint) -> None:
        if not isinstance(print_, PricePrint):
            raise TypeError("print must be PricePrint")
        self._prints.append(print_)
        self._prints.sort(key=lambda item: (item.knowledge_time, item.symbol))

    def next_eligible(self, symbol: str, *, after: datetime) -> PricePrint | None:
        symbol = symbol.strip().upper()
        after = _aware(after, "after")
        for item in self._prints:
            if item.symbol == symbol and item.knowledge_time > after:
                return item
        return None

    def prices_as_of(self, when: datetime) -> dict[str, float]:
        when = _aware(when, "when")
        latest: dict[str, PricePrint] = {}
        for item in self._prints:
            if item.knowledge_time <= when:
                latest[item.symbol] = item
        return {symbol: item.price for symbol, item in latest.items()}

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
    ) -> Fill | None:
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if not isinstance(tape, MarketTape):
            raise TypeError("tape must be MarketTape")
        if candidate.status is not TradeCandidateStatus.APPROVED:
            raise ExecutionError("only approved candidates can be submitted")
        if candidate.direction is TradeSide.NO_TRADE:
            return None
        equity_value = _finite(equity, "equity")
        if equity_value <= 0:
            raise ExecutionError("equity must be positive")
        nxt = tape.next_eligible(candidate.instrument, after=candidate.knowledge_time)
        if nxt is None:
            return None
        reference = Decimal(str(nxt.price))
        notional = Decimal(str(equity_value)) * Decimal(str(candidate.proposed_size))
        quantity = notional / reference
        if quantity <= 0:
            return None
        slip = quantity * reference * Decimal(str(self.config.slippage_bps)) / Decimal("10000")
        fee = quantity * reference * Decimal(str(self.config.fee_bps)) / Decimal("10000")
        side = FillSide.BUY if candidate.direction is TradeSide.LONG else FillSide.SELL
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
