from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import fsum, isfinite, sqrt
from types import MappingProxyType

from .portfolio import PortfolioSnapshot
from .types import TradeCandidate, TradeSide


class PortfolioRiskError(ValueError):
    """Portfolio-wide risk inputs are missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class PortfolioRiskInput:
    portfolio: PortfolioSnapshot
    sectors: Mapping[str, str]
    factor_loadings: Mapping[str, Mapping[str, float]]
    annualized_volatility: Mapping[str, float]
    correlations: Mapping[str, Mapping[str, float]]
    current_drawdown: float

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise TypeError("portfolio must be PortfolioSnapshot")
        for name in (
            "sectors", "factor_loadings", "annualized_volatility", "correlations"
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        drawdown = _finite(self.current_drawdown, "current_drawdown")
        if not 0 <= drawdown <= 1:
            raise PortfolioRiskError("current_drawdown must be between 0 and 1")
        object.__setattr__(self, "current_drawdown", drawdown)


@dataclass(frozen=True, slots=True)
class PortfolioRiskSnapshot:
    gross_exposure: float
    net_exposure: float
    sector_exposure: Mapping[str, float]
    factor_exposure: Mapping[str, float]
    annualized_volatility: float
    concentration: float
    drawdown: float
    correlation_clusters: tuple[tuple[str, ...], ...]
    knowledge_time: datetime


@dataclass(frozen=True, slots=True)
class PortfolioRiskComparison:
    before: PortfolioRiskSnapshot
    after: PortfolioRiskSnapshot
    trade_id: str | None


class PortfolioRiskCalculator:
    """Pure deterministic before/after portfolio risk calculations."""

    def __init__(self, *, cluster_threshold: float = 0.75) -> None:
        threshold = _finite(cluster_threshold, "cluster_threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("cluster_threshold must be between 0 and 1")
        self.cluster_threshold = threshold

    def compare(
        self,
        inputs: PortfolioRiskInput,
        candidate: TradeCandidate,
        *,
        approved_size: float | None = None,
    ) -> PortfolioRiskComparison:
        if not isinstance(inputs, PortfolioRiskInput):
            raise TypeError("inputs must be PortfolioRiskInput")
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        size = candidate.proposed_size if approved_size is None else _finite(
            approved_size, "approved_size"
        )
        if size < 0 or size > candidate.proposed_size:
            raise PortfolioRiskError(
                "approved_size must be between zero and the proposed size"
            )
        before_values = self._position_values(inputs)
        after_values = dict(before_values)
        if candidate.direction is not TradeSide.NO_TRADE and size:
            change = size * inputs.portfolio.equity
            if candidate.direction is TradeSide.SHORT:
                change = -change
            symbol = candidate.instrument
            after_values[symbol] = after_values.get(symbol, 0.0) + change
            if abs(after_values[symbol]) < 1e-12:
                after_values.pop(symbol)
        return PortfolioRiskComparison(
            before=self._calculate(inputs, before_values),
            after=self._calculate(inputs, after_values),
            trade_id=candidate.id.value,
        )

    def snapshot(self, inputs: PortfolioRiskInput) -> PortfolioRiskSnapshot:
        if not isinstance(inputs, PortfolioRiskInput):
            raise TypeError("inputs must be PortfolioRiskInput")
        return self._calculate(inputs, self._position_values(inputs))

    @staticmethod
    def _position_values(inputs: PortfolioRiskInput) -> dict[str, float]:
        equity = _finite(inputs.portfolio.equity, "portfolio.equity")
        if equity <= 0:
            raise PortfolioRiskError("portfolio equity must be positive")
        values: dict[str, float] = {}
        for position in inputs.portfolio.positions:
            if position.market_price is None:
                raise PortfolioRiskError(
                    f"missing market price for {position.instrument}"
                )
            price = _finite(position.market_price, "market_price")
            quantity = _finite(position.quantity, "quantity")
            if price <= 0:
                raise PortfolioRiskError("market prices must be positive")
            symbol = position.instrument.strip().upper()
            values[symbol] = values.get(symbol, 0.0) + quantity * price
        return {key: value for key, value in values.items() if value != 0}

    def _calculate(
        self, inputs: PortfolioRiskInput, values: Mapping[str, float]
    ) -> PortfolioRiskSnapshot:
        equity = float(inputs.portfolio.equity)
        weights = {symbol: value / equity for symbol, value in values.items()}
        self._validate_metadata(inputs, tuple(weights))
        sector: dict[str, float] = {}
        factor: dict[str, float] = {}
        for symbol, weight in weights.items():
            sector_name = inputs.sectors[symbol].strip()
            sector[sector_name] = sector.get(sector_name, 0.0) + weight
            for factor_name, loading in inputs.factor_loadings[symbol].items():
                name = factor_name.strip()
                factor[name] = factor.get(name, 0.0) + weight * float(loading)
        variance = 0.0
        symbols = tuple(sorted(weights))
        for left in symbols:
            for right in symbols:
                correlation = 1.0 if left == right else _correlation(
                    inputs.correlations, left, right
                )
                variance += (
                    weights[left]
                    * weights[right]
                    * float(inputs.annualized_volatility[left])
                    * float(inputs.annualized_volatility[right])
                    * correlation
                )
        return PortfolioRiskSnapshot(
            gross_exposure=fsum(abs(value) for value in weights.values()),
            net_exposure=fsum(weights.values()),
            sector_exposure=MappingProxyType(dict(sorted(sector.items()))),
            factor_exposure=MappingProxyType(dict(sorted(factor.items()))),
            annualized_volatility=sqrt(max(variance, 0.0)),
            concentration=max((abs(value) for value in weights.values()), default=0.0),
            drawdown=inputs.current_drawdown,
            correlation_clusters=self._clusters(inputs.correlations, symbols),
            knowledge_time=inputs.portfolio.knowledge_time,
        )

    @staticmethod
    def _validate_metadata(
        inputs: PortfolioRiskInput, symbols: tuple[str, ...]
    ) -> None:
        for symbol in symbols:
            sector = inputs.sectors.get(symbol)
            if not isinstance(sector, str) or not sector.strip():
                raise PortfolioRiskError(f"missing sector for {symbol}")
            loadings = inputs.factor_loadings.get(symbol)
            if not isinstance(loadings, Mapping) or not loadings:
                raise PortfolioRiskError(f"missing factor loadings for {symbol}")
            for name, value in loadings.items():
                if not isinstance(name, str) or not name.strip():
                    raise PortfolioRiskError("factor names must be non-empty")
                _finite(value, f"factor loading {symbol}.{name}")
            volatility = inputs.annualized_volatility.get(symbol)
            if volatility is None or _finite(volatility, f"volatility {symbol}") < 0:
                raise PortfolioRiskError(f"missing or invalid volatility for {symbol}")
        for index, left in enumerate(symbols):
            for right in symbols[index + 1 :]:
                _correlation(inputs.correlations, left, right)

    def _clusters(
        self,
        correlations: Mapping[str, Mapping[str, float]],
        symbols: tuple[str, ...],
    ) -> tuple[tuple[str, ...], ...]:
        remaining = set(symbols)
        clusters: list[tuple[str, ...]] = []
        while remaining:
            seed = min(remaining)
            component = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                left = frontier.pop()
                linked = {
                    right
                    for right in tuple(remaining)
                    if abs(_correlation(correlations, left, right))
                    >= self.cluster_threshold
                }
                remaining -= linked
                component |= linked
                frontier.extend(sorted(linked))
            if len(component) > 1:
                clusters.append(tuple(sorted(component)))
        return tuple(sorted(clusters))


def _correlation(
    correlations: Mapping[str, Mapping[str, float]], left: str, right: str
) -> float:
    raw = correlations.get(left, {}).get(right)
    if raw is None:
        raw = correlations.get(right, {}).get(left)
    if raw is None:
        raise PortfolioRiskError(f"missing correlation for {left}/{right}")
    value = _finite(raw, f"correlation {left}/{right}")
    if not -1 <= value <= 1:
        raise PortfolioRiskError("correlations must be between -1 and 1")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PortfolioRiskError(f"{name} must be numeric")
    number = float(value)
    if not isfinite(number):
        raise PortfolioRiskError(f"{name} must be finite")
    return number
