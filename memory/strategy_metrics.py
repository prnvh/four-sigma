from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite, sqrt
from statistics import mean, stdev

from .portfolio import Fill, FillSide, PortfolioSnapshot


_YEAR_SECONDS = 365.25 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    total_return: float
    cagr: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    hit_rate: float | None
    profit_factor: float | None
    turnover: float
    transaction_cost: float
    exposure: float
    monthly_returns: tuple[tuple[str, float], ...]
    positive_month_rate: float | None
    monthly_return_volatility: float | None


def calculate_strategy_metrics(
    snapshots: Sequence[PortfolioSnapshot], fills: Sequence[Fill]
) -> StrategyMetrics:
    """Calculate deterministic strategy metrics from a completed portfolio trace."""
    if not snapshots:
        raise ValueError("strategy metrics require at least one portfolio snapshot")
    if any(not isinstance(item, PortfolioSnapshot) for item in snapshots):
        raise TypeError("snapshots must contain only PortfolioSnapshot values")
    if any(not isinstance(item, Fill) for item in fills):
        raise TypeError("fills must contain only Fill values")
    if any(
        later.knowledge_time < earlier.knowledge_time
        for earlier, later in zip(snapshots, snapshots[1:])
    ):
        raise ValueError("snapshots must be in knowledge_time order")
    if any(
        later.knowledge_time < earlier.knowledge_time
        for earlier, later in zip(fills, fills[1:])
    ):
        raise ValueError("fills must be in knowledge_time order")

    equities = [float(item.equity) for item in snapshots]
    if any(not isfinite(item) for item in equities) or equities[0] <= 0:
        raise ValueError("snapshot equity must be finite and initial equity must be positive")
    total_return = equities[-1] / equities[0] - 1
    elapsed = (snapshots[-1].knowledge_time - snapshots[0].knowledge_time).total_seconds()
    cagr = None
    if elapsed > 0 and equities[-1] > 0:
        try:
            cagr = (equities[-1] / equities[0]) ** (_YEAR_SECONDS / elapsed) - 1
        except OverflowError:
            cagr = None

    returns = [
        current / previous - 1
        for previous, current in zip(equities, equities[1:])
        if previous > 0
    ]
    periods_per_year = len(returns) * _YEAR_SECONDS / elapsed if elapsed > 0 else 0
    sharpe = None
    if len(returns) >= 2 and periods_per_year > 0:
        volatility = stdev(returns)
        if volatility > 0:
            sharpe = mean(returns) / volatility * sqrt(periods_per_year)
    sortino = None
    if returns and periods_per_year > 0:
        downside = sqrt(mean(min(item, 0.0) ** 2 for item in returns))
        if downside > 0:
            sortino = mean(returns) / downside * sqrt(periods_per_year)

    peak = equities[0]
    max_drawdown = 0.0
    for equity in equities:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)

    outcomes = _closed_trade_outcomes(fills)
    hit_rate = None if not outcomes else sum(item > 0 for item in outcomes) / len(outcomes)
    gross_profit = sum((item for item in outcomes if item > 0), Decimal("0"))
    gross_loss = -sum((item for item in outcomes if item < 0), Decimal("0"))
    profit_factor = None if gross_loss == 0 else float(gross_profit / gross_loss)

    transaction_cost = sum(item.fee + item.slippage for item in fills)
    traded_notional = sum(item.quantity * item.price for item in fills)
    average_equity = sum(equities) / len(equities)
    turnover = 0.0 if average_equity <= 0 else traded_notional / average_equity

    exposure_points = [_snapshot_exposure(item) for item in snapshots]
    if elapsed > 0:
        weighted = sum(
            exposure_points[index]
            * (snapshots[index + 1].knowledge_time - snapshot.knowledge_time).total_seconds()
            for index, snapshot in enumerate(snapshots[:-1])
        )
        exposure = weighted / elapsed
    else:
        exposure = exposure_points[-1]

    month_ends: dict[str, float] = {}
    for snapshot in snapshots:
        month_ends[snapshot.knowledge_time.strftime("%Y-%m")] = float(snapshot.equity)
    monthly: list[tuple[str, float]] = []
    baseline = equities[0]
    for month, ending_equity in month_ends.items():
        monthly.append((month, ending_equity / baseline - 1))
        baseline = ending_equity
    positive_month_rate = (
        None
        if not monthly
        else sum(value > 0 for _, value in monthly) / len(monthly)
    )
    monthly_return_volatility = (
        stdev(value for _, value in monthly) if len(monthly) >= 2 else None
    )

    return StrategyMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        profit_factor=profit_factor,
        turnover=turnover,
        transaction_cost=transaction_cost,
        exposure=exposure,
        monthly_returns=tuple(monthly),
        positive_month_rate=positive_month_rate,
        monthly_return_volatility=monthly_return_volatility,
    )


def _snapshot_exposure(snapshot: PortfolioSnapshot) -> float:
    if snapshot.equity <= 0:
        return 0.0
    gross = sum(
        abs(position.quantity)
        * (position.market_price if position.market_price is not None else position.average_entry)
        for position in snapshot.positions
    )
    return gross / snapshot.equity


def _closed_trade_outcomes(fills: Sequence[Fill]) -> tuple[Decimal, ...]:
    positions: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
    outcomes: list[Decimal] = []
    for fill in fills:
        quantity = Decimal(str(fill.quantity))
        price = Decimal(str(fill.price))
        cost = Decimal(str(fill.fee + fill.slippage))
        signed = quantity if fill.side is FillSide.BUY else -quantity
        old_quantity, average, cycle_pnl = positions.get(
            fill.instrument, (Decimal("0"), Decimal("0"), Decimal("0"))
        )
        if old_quantity == 0:
            positions[fill.instrument] = (signed, price, -cost)
            continue
        if (old_quantity > 0) == (signed > 0):
            new_quantity = old_quantity + signed
            new_average = (
                abs(old_quantity) * average + quantity * price
            ) / abs(new_quantity)
            positions[fill.instrument] = (new_quantity, new_average, cycle_pnl - cost)
            continue

        closed = min(abs(old_quantity), quantity)
        closing_cost = cost * closed / quantity
        realized = (
            (price - average) * closed
            if old_quantity > 0
            else (average - price) * closed
        )
        cycle_pnl += realized - closing_cost
        new_quantity = old_quantity + signed
        if new_quantity == 0:
            outcomes.append(cycle_pnl)
            positions.pop(fill.instrument)
        elif (old_quantity > 0) == (new_quantity > 0):
            positions[fill.instrument] = (new_quantity, average, cycle_pnl)
        else:
            outcomes.append(cycle_pnl)
            positions[fill.instrument] = (new_quantity, price, -(cost - closing_cost))
    return tuple(outcomes)
