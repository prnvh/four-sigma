from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from math import isfinite
from types import MappingProxyType

from .audit_logger import AuditEventType, AuditLedger
from .capabilities import Action, CAPABILITIES
from .portfolio import PortfolioSnapshot
from .types import AgentId, AuditEventId, CreatedAt, RunId, TradeCandidate, TradeSide


class RiskCheckResult(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    RESIZE = "RESIZE"


class RiskReasonCode(StrEnum):
    MAX_POSITION = "max_position_pct"
    GROSS_EXPOSURE = "gross_exposure"
    NET_EXPOSURE = "net_exposure"
    SECTOR_CONCENTRATION = "sector_concentration"
    SINGLE_NAME_CONCENTRATION = "single_name_concentration"
    DAILY_LIQUIDITY = "daily_liquidity"
    VOLATILITY_LIMIT = "volatility_limit"
    DRAWDOWN_CONSTRAINT = "drawdown_constraint"
    INVALID_EQUITY = "invalid_equity"
    MISSING_MARKET_PRICE = "missing_market_price"
    MISSING_SECTOR = "missing_sector"
    MISSING_DAILY_LIQUIDITY = "missing_daily_liquidity"
    MISSING_VOLATILITY = "missing_volatility"


@dataclass(frozen=True, slots=True)
class RiskReason:
    code: RiskReasonCode
    actual: float | None = None
    limit: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code.value, "actual": self.actual, "limit": self.limit}


@dataclass(frozen=True, slots=True)
class PositionRiskDecision:
    result: RiskCheckResult
    requested_size: float
    approved_size: float
    reasons: tuple[RiskReason, ...]

    @property
    def vetoed(self) -> bool:
        return self.result is RiskCheckResult.REJECT


@dataclass(frozen=True, slots=True)
class PositionRiskLimits:
    max_position_pct: float = 0.10
    max_gross_exposure: float = 1.50
    max_net_exposure: float = 0.50
    max_sector_concentration: float = 0.30
    max_single_name_concentration: float = 0.15
    max_daily_liquidity_pct: float = 0.10
    max_annualized_volatility: float = 0.60
    max_drawdown: float = 0.20

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        positive = (
            self.max_position_pct,
            self.max_gross_exposure,
            self.max_sector_concentration,
            self.max_single_name_concentration,
            self.max_daily_liquidity_pct,
            self.max_annualized_volatility,
        )
        if any(value == 0 for value in positive):
            raise ValueError("position, exposure, liquidity and volatility limits must be positive")


@dataclass(frozen=True, slots=True)
class PositionRiskInput:
    candidate: TradeCandidate
    portfolio: PortfolioSnapshot
    sectors: Mapping[str, str]
    average_daily_dollar_volume: Mapping[str, float]
    annualized_volatility: Mapping[str, float]
    current_drawdown: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise TypeError("portfolio must be PortfolioSnapshot")
        for name in ("sectors", "average_daily_dollar_volume", "annualized_volatility"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            normalized = {
                key.strip().upper(): item
                for key, item in value.items()
                if isinstance(key, str) and key.strip()
            }
            object.__setattr__(self, name, MappingProxyType(normalized))


def _number(value: object) -> Decimal | None:
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


class DeterministicPositionRiskEngine:
    """Read-only pre-trade limits. The same inputs always yield the same decision."""

    def __init__(self, limits: PositionRiskLimits, audit_ledger: AuditLedger) -> None:
        if not isinstance(limits, PositionRiskLimits):
            raise TypeError("limits must be PositionRiskLimits")
        if not isinstance(audit_ledger, AuditLedger):
            raise TypeError("audit_ledger must be AuditLedger")
        self.limits = limits
        self.audit_ledger = audit_ledger

    def evaluate(
        self,
        inputs: PositionRiskInput,
        *,
        audit_event_id: AuditEventId,
        run_id: RunId | None = None,
    ) -> PositionRiskDecision:
        if not isinstance(inputs, PositionRiskInput):
            raise TypeError("inputs must be PositionRiskInput")
        CAPABILITIES.require("portfolio_risk", Action.READ, "portfolio", "positions")
        CAPABILITIES.require("portfolio_risk", Action.VETO, "trade_candidates", "*")

        if inputs.candidate.direction is TradeSide.NO_TRADE:
            decision = PositionRiskDecision(RiskCheckResult.PASS, 0.0, 0.0, ())
        else:
            decision = self._decide(inputs)
        self.audit_ledger.append(
            event_id=audit_event_id,
            event_type=AuditEventType.RISK_CHECK_RUN,
            occurred_at=CreatedAt(inputs.candidate.knowledge_time),
            details={
                "result": decision.result.value,
                "requested_size": decision.requested_size,
                "approved_size": decision.approved_size,
                "reasons": [reason.as_dict() for reason in decision.reasons],
            },
            agent_id=AgentId("portfolio_risk"),
            run_id=run_id,
            subject_id=inputs.candidate.id,
        )
        return decision

    def _decide(self, inputs: PositionRiskInput) -> PositionRiskDecision:
        requested = Decimal(str(inputs.candidate.proposed_size))
        hard_reasons, positions = self._hard_reasons(inputs)
        if hard_reasons:
            return PositionRiskDecision(
                RiskCheckResult.REJECT, float(requested), 0.0, tuple(hard_reasons)
            )

        requested_metrics = self._metrics(inputs, positions, requested)
        sizing_reasons = self._sizing_reasons(requested_metrics)
        if not sizing_reasons:
            return PositionRiskDecision(
                RiskCheckResult.PASS, float(requested), float(requested), ()
            )

        safe = self._maximum_safe_size(inputs, positions, requested)
        result = RiskCheckResult.RESIZE if safe > 0 else RiskCheckResult.REJECT
        return PositionRiskDecision(
            result,
            float(requested),
            round(float(safe), 12) if safe > 0 else 0.0,
            tuple(sizing_reasons),
        )

    def _hard_reasons(
        self, inputs: PositionRiskInput
    ) -> tuple[list[RiskReason], dict[str, Decimal]]:
        reasons: list[RiskReason] = []
        equity = _number(inputs.portfolio.equity)
        if equity is None or equity <= 0:
            reasons.append(RiskReason(RiskReasonCode.INVALID_EQUITY))

        positions: dict[str, Decimal] = {}
        for position in inputs.portfolio.positions:
            price = _number(position.market_price)
            quantity = _number(position.quantity)
            if price is None or price <= 0 or quantity is None:
                reasons.append(RiskReason(RiskReasonCode.MISSING_MARKET_PRICE))
                continue
            positions[position.instrument.strip().upper()] = quantity * price

        symbols = set(positions) | {inputs.candidate.instrument}
        if any(not isinstance(inputs.sectors.get(symbol), str) or not inputs.sectors[symbol].strip() for symbol in symbols):
            reasons.append(RiskReason(RiskReasonCode.MISSING_SECTOR))
        liquidity = _number(
            inputs.average_daily_dollar_volume.get(inputs.candidate.instrument)
        )
        if liquidity is None or liquidity <= 0:
            reasons.append(RiskReason(RiskReasonCode.MISSING_DAILY_LIQUIDITY))
        volatility = _number(inputs.annualized_volatility.get(inputs.candidate.instrument))
        if volatility is None or volatility < 0:
            reasons.append(RiskReason(RiskReasonCode.MISSING_VOLATILITY))
        elif volatility > Decimal(str(self.limits.max_annualized_volatility)):
            reasons.append(
                RiskReason(
                    RiskReasonCode.VOLATILITY_LIMIT,
                    float(volatility),
                    self.limits.max_annualized_volatility,
                )
            )
        drawdown = _number(inputs.current_drawdown)
        if drawdown is None or drawdown < 0:
            reasons.append(RiskReason(RiskReasonCode.DRAWDOWN_CONSTRAINT))
        elif drawdown > Decimal(str(self.limits.max_drawdown)):
            reasons.append(
                RiskReason(
                    RiskReasonCode.DRAWDOWN_CONSTRAINT,
                    float(drawdown),
                    self.limits.max_drawdown,
                )
            )
        return reasons, positions

    def _metrics(
        self,
        inputs: PositionRiskInput,
        current: Mapping[str, Decimal],
        size: Decimal,
    ) -> dict[RiskReasonCode, Decimal]:
        equity = Decimal(str(inputs.portfolio.equity))
        signed_trade = size * equity
        if inputs.candidate.direction is TradeSide.SHORT:
            signed_trade = -signed_trade
        after = dict(current)
        symbol = inputs.candidate.instrument
        after[symbol] = after.get(symbol, Decimal("0")) + signed_trade
        gross = sum((abs(value) for value in after.values()), Decimal("0")) / equity
        net = abs(sum(after.values(), Decimal("0"))) / equity
        sector = inputs.sectors[symbol].strip()
        sector_value = sum(
            abs(value)
            for item, value in after.items()
            if inputs.sectors[item].strip() == sector
        )
        liquidity = Decimal(
            str(inputs.average_daily_dollar_volume[symbol])
        )
        return {
            RiskReasonCode.MAX_POSITION: size,
            RiskReasonCode.GROSS_EXPOSURE: gross,
            RiskReasonCode.NET_EXPOSURE: net,
            RiskReasonCode.SECTOR_CONCENTRATION: sector_value / equity,
            RiskReasonCode.SINGLE_NAME_CONCENTRATION: abs(after[symbol]) / equity,
            RiskReasonCode.DAILY_LIQUIDITY: abs(signed_trade) / liquidity,
        }

    def _sizing_reasons(
        self, metrics: Mapping[RiskReasonCode, Decimal]
    ) -> list[RiskReason]:
        limits = {
            RiskReasonCode.MAX_POSITION: self.limits.max_position_pct,
            RiskReasonCode.GROSS_EXPOSURE: self.limits.max_gross_exposure,
            RiskReasonCode.NET_EXPOSURE: self.limits.max_net_exposure,
            RiskReasonCode.SECTOR_CONCENTRATION: self.limits.max_sector_concentration,
            RiskReasonCode.SINGLE_NAME_CONCENTRATION: self.limits.max_single_name_concentration,
            RiskReasonCode.DAILY_LIQUIDITY: self.limits.max_daily_liquidity_pct,
        }
        return [
            RiskReason(code, float(metrics[code]), limit)
            for code, limit in limits.items()
            if metrics[code] > Decimal(str(limit))
        ]

    def _maximum_safe_size(
        self,
        inputs: PositionRiskInput,
        current: Mapping[str, Decimal],
        requested: Decimal,
    ) -> Decimal:
        if self._sizing_reasons(self._metrics(inputs, current, Decimal("0"))):
            return Decimal("0")
        low, high = Decimal("0"), requested
        for _ in range(64):
            middle = (low + high) / 2
            if self._sizing_reasons(self._metrics(inputs, current, middle)):
                high = middle
            else:
                low = middle
        return low
