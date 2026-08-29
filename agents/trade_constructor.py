from __future__ import annotations

from collections.abc import Sequence

from memory.capabilities import CAPABILITIES
from memory.context_gateway import TradeConstructionContext
from memory.types import (
    Direction,
    InsightId,
    PromotedInsight,
    TradeCandidate,
    TradeCandidateId,
    TradeSide,
)

from .registry import TRADE_CONSTRUCTOR_V1, AgentSpec


def _default_size(spec: AgentSpec) -> float:
    size = spec.config.get("default_size", 0.01)
    if isinstance(size, bool) or not isinstance(size, (int, float)) or size <= 0:
        raise ValueError("trade constructor default_size must be a positive number")
    return float(size)


def _horizon(spec: AgentSpec) -> str:
    horizon = spec.config.get("horizon", "from_approved_insights")
    if not isinstance(horizon, str) or not horizon.strip():
        raise ValueError("trade constructor horizon must be a non-empty string")
    return horizon.strip()


def _causing_insights(
    insights: tuple[PromotedInsight, ...],
) -> tuple[TradeSide, tuple[PromotedInsight, ...]]:
    bullish = tuple(item for item in insights if item.direction is Direction.BULLISH)
    bearish = tuple(item for item in insights if item.direction is Direction.BEARISH)
    if len(bullish) > len(bearish):
        return TradeSide.LONG, bullish
    if len(bearish) > len(bullish):
        return TradeSide.SHORT, bearish
    return TradeSide.NO_TRADE, insights


class TradeConstructor:
    """Maps approved insights to a candidate. Deterministic. Never reads raw news."""

    def __init__(self, spec: AgentSpec = TRADE_CONSTRUCTOR_V1) -> None:
        if spec.name != "trade_constructor":
            raise ValueError("TradeConstructor requires a trade_constructor spec")
        self.spec = spec

    def propose(
        self,
        context: TradeConstructionContext,
        *,
        requested_fields: Sequence[tuple[str, str]] = (),
    ) -> TradeCandidate:
        if not isinstance(context, TradeConstructionContext):
            raise TypeError("TradeConstructor requires context from ContextGateway")
        if hasattr(context, "articles"):
            raise ValueError("trade construction cannot use raw news")
        CAPABILITIES.require_reads(
            self.spec.name, (("insights", "promoted"), *requested_fields)
        )
        insights = context.promoted_insights
        if not insights:
            raise ValueError("trade construction requires at least one approved insight")
        side, causing = _causing_insights(insights)
        confidence = sum(item.confidence for item in causing) / len(causing)
        size = 0.0 if side is TradeSide.NO_TRADE else _default_size(self.spec)
        if side is TradeSide.LONG:
            entry = ("majority of approved insights are bullish",)
            exit_ = ("cited bullish insights expire or are superseded",)
        elif side is TradeSide.SHORT:
            entry = ("majority of approved insights are bearish",)
            exit_ = ("cited bearish insights expire or are superseded",)
        else:
            entry = ()
            exit_ = ("wait until approved insights agree on direction",)
        return TradeCandidate(
            id=TradeCandidateId(
                f"{context.symbol}:{context.simulation_time.isoformat()}:{side.value}"
            ),
            instrument=context.symbol,
            direction=side,
            thesis_refs=tuple(InsightId(item.ref) for item in causing),
            horizon=_horizon(self.spec),
            confidence=confidence,
            entry_conditions=entry,
            exit_conditions=exit_,
            proposed_size=size,
            knowledge_time=context.simulation_time,
        )
