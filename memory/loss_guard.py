from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite

from .portfolio import PortfolioSnapshot
from .types import TradeCandidate, TradeSide


class LossGuardReason(StrEnum):
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    POSITION_STOP_LOSS = "position_stop_loss"
    AVERAGING_DOWN = "averaging_down"
    LOSS_COOLDOWN = "loss_cooldown"


@dataclass(frozen=True, slots=True)
class LossGuardLimits:
    max_portfolio_drawdown: float = 0.10
    max_position_loss: float = 0.08
    max_consecutive_losses: int = 1
    loss_cooldown: timedelta = timedelta(days=5)
    allow_averaging_down: bool = False

    def __post_init__(self) -> None:
        for name in ("max_portfolio_drawdown", "max_position_loss"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ValueError(f"{name} must be greater than 0 and at most 1")
            object.__setattr__(self, name, float(value))
        if (
            isinstance(self.max_consecutive_losses, bool)
            or not isinstance(self.max_consecutive_losses, int)
            or self.max_consecutive_losses < 1
        ):
            raise ValueError("max_consecutive_losses must be a positive integer")
        if not isinstance(self.loss_cooldown, timedelta) or self.loss_cooldown <= timedelta(0):
            raise ValueError("loss_cooldown must be a positive timedelta")
        if not isinstance(self.allow_averaging_down, bool):
            raise TypeError("allow_averaging_down must be bool")


@dataclass(frozen=True, slots=True)
class LossGuardDecision:
    allowed: bool
    must_exit: bool
    reasons: tuple[LossGuardReason, ...]
    portfolio_drawdown: float
    position_loss: float


@dataclass(slots=True)
class _LossState:
    consecutive_losses: int = 0
    last_loss_at: datetime | None = None


class DeterministicLossGuard:
    """Stateful hard veto against loss chasing and repeated losing re-entry."""

    def __init__(self, limits: LossGuardLimits | None = None) -> None:
        self.limits = limits or LossGuardLimits()
        if not isinstance(self.limits, LossGuardLimits):
            raise TypeError("limits must be LossGuardLimits")
        self._states: dict[str, _LossState] = {}

    def record_closed_trade(
        self, instrument: str, realized_pnl: float, *, closed_at: datetime
    ) -> None:
        symbol = self._symbol(instrument)
        pnl = self._finite(realized_pnl, "realized_pnl")
        self._aware(closed_at, "closed_at")
        state = self._states.setdefault(symbol, _LossState())
        if pnl < 0:
            state.consecutive_losses += 1
            state.last_loss_at = closed_at
        elif pnl > 0:
            state.consecutive_losses = 0
            state.last_loss_at = None

    def evaluate(
        self,
        candidate: TradeCandidate,
        portfolio: PortfolioSnapshot,
        *,
        peak_equity: float,
        evaluated_at: datetime,
    ) -> LossGuardDecision:
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if not isinstance(portfolio, PortfolioSnapshot):
            raise TypeError("portfolio must be PortfolioSnapshot")
        peak = self._finite(peak_equity, "peak_equity")
        self._aware(evaluated_at, "evaluated_at")
        if peak <= 0:
            raise ValueError("peak_equity must be positive")
        drawdown = max(0.0, 1.0 - portfolio.equity / peak)
        position_loss = 0.0
        position = next(
            (item for item in portfolio.positions if item.instrument == candidate.instrument),
            None,
        )
        same_exposure = False
        if position is not None and position.market_price is not None and position.average_entry > 0:
            if position.quantity > 0:
                position_loss = max(
                    0.0, (position.average_entry - position.market_price) / position.average_entry
                )
                same_exposure = candidate.direction is TradeSide.LONG
            elif position.quantity < 0:
                position_loss = max(
                    0.0, (position.market_price - position.average_entry) / position.average_entry
                )
                same_exposure = candidate.direction is TradeSide.SHORT

        reasons: list[LossGuardReason] = []
        must_exit = False
        if drawdown >= self.limits.max_portfolio_drawdown:
            reasons.append(LossGuardReason.PORTFOLIO_DRAWDOWN)
            must_exit = position is not None
        if position_loss >= self.limits.max_position_loss:
            reasons.append(LossGuardReason.POSITION_STOP_LOSS)
            must_exit = True
        if (
            position_loss > 0
            and same_exposure
            and not self.limits.allow_averaging_down
        ):
            reasons.append(LossGuardReason.AVERAGING_DOWN)
        state = self._states.get(candidate.instrument)
        if (
            state is not None
            and state.consecutive_losses >= self.limits.max_consecutive_losses
            and state.last_loss_at is not None
            and evaluated_at < state.last_loss_at + self.limits.loss_cooldown
        ):
            reasons.append(LossGuardReason.LOSS_COOLDOWN)
        unique = tuple(dict.fromkeys(reasons))
        return LossGuardDecision(
            allowed=not unique,
            must_exit=must_exit,
            reasons=unique,
            portfolio_drawdown=drawdown,
            position_loss=position_loss,
        )

    def required_exits(
        self, portfolio: PortfolioSnapshot, *, peak_equity: float
    ) -> dict[str, tuple[LossGuardReason, ...]]:
        """Return hard exits that must be submitted without waiting for an agent."""

        if not isinstance(portfolio, PortfolioSnapshot):
            raise TypeError("portfolio must be PortfolioSnapshot")
        peak = self._finite(peak_equity, "peak_equity")
        if peak <= 0:
            raise ValueError("peak_equity must be positive")
        drawdown = max(0.0, 1.0 - portfolio.equity / peak)
        exits: dict[str, tuple[LossGuardReason, ...]] = {}
        for position in portfolio.positions:
            reasons: list[LossGuardReason] = []
            if drawdown >= self.limits.max_portfolio_drawdown:
                reasons.append(LossGuardReason.PORTFOLIO_DRAWDOWN)
            loss = 0.0
            if position.market_price is not None and position.average_entry > 0:
                if position.quantity > 0:
                    loss = max(
                        0.0,
                        (position.average_entry - position.market_price)
                        / position.average_entry,
                    )
                elif position.quantity < 0:
                    loss = max(
                        0.0,
                        (position.market_price - position.average_entry)
                        / position.average_entry,
                    )
            if loss >= self.limits.max_position_loss:
                reasons.append(LossGuardReason.POSITION_STOP_LOSS)
            if reasons:
                exits[position.instrument] = tuple(reasons)
        return exits

    def consecutive_losses(self, instrument: str) -> int:
        state = self._states.get(self._symbol(instrument))
        return 0 if state is None else state.consecutive_losses

    @staticmethod
    def _symbol(value: str) -> str:
        symbol = value.strip().upper() if isinstance(value, str) else ""
        if not symbol:
            raise ValueError("instrument must be non-empty")
        return symbol

    @staticmethod
    def _finite(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        number = float(value)
        if not isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number

    @staticmethod
    def _aware(value: datetime, name: str) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError(f"{name} must be datetime")
        if value.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        return value
