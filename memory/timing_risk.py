from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from statistics import median

from .execution import MarketTape
from .types import (
    Evidence,
    PromotedInsight,
    TradeCandidate,
    TradeCandidateStatus,
    TradeSide,
)


class TimingAction(StrEnum):
    ALLOW = "allow"
    DEFER = "defer"
    REDUCE = "reduce"


@dataclass(frozen=True, slots=True)
class TapeFacts:
    last_price: float | None
    window_start_price: float | None
    high_20d: float | None
    return_20d: float | None
    distance_from_high: float | None


@dataclass(frozen=True, slots=True)
class TimingRiskDecision:
    action: TimingAction
    candidate: TradeCandidate
    facts: TapeFacts
    reasons: tuple[str, ...]


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def tape_facts(tape: MarketTape, symbol: str, now: datetime) -> TapeFacts:
    now = _aware(now, "now")
    if not isinstance(tape, MarketTape):
        raise TypeError("tape must be MarketTape")
    start = now - timedelta(days=20)
    window = [
        item
        for item in tape.prints
        if item.symbol == symbol and start <= item.knowledge_time <= now
    ]
    empty = TapeFacts(None, None, None, None, None)
    if len(window) < 2:
        return empty
    first = window[0].price
    last = window[-1].price
    high = max(item.price for item in window)
    if first <= 0 or last <= 0 or high <= 0:
        return empty
    return TapeFacts(
        last_price=last,
        window_start_price=first,
        high_20d=high,
        return_20d=last / first - 1,
        distance_from_high=(high - last) / high,
    )


@dataclass(frozen=True, slots=True)
class TapeReaction:
    """Whether the tape moved enough vs this name's own recent days to look again."""

    triggered: bool
    last_price: float | None
    reference_price: float | None
    move: float | None
    typical_daily_move: float | None


_REACTION_MULTIPLE = 2.0
_MIN_DAILY_RETURNS = 5


def tape_reaction(
    tape: MarketTape,
    symbol: str,
    now: datetime,
    *,
    reference_price: float | None = None,
) -> TapeReaction:
    now = _aware(now, "now")
    if not isinstance(tape, MarketTape):
        raise TypeError("tape must be MarketTape")
    if reference_price is not None:
        if (
            isinstance(reference_price, bool)
            or not isinstance(reference_price, (int, float))
            or not isfinite(reference_price)
            or reference_price <= 0
        ):
            raise ValueError("reference_price must be a positive number")
    closes: dict[object, float] = {}
    last = None
    for item in tape.prints:
        if item.symbol != symbol or item.knowledge_time > now:
            continue
        last = item.price
        closes[item.knowledge_time.date()] = item.price
    empty = TapeReaction(False, last, None, None, None)
    if last is None:
        return empty
    days = sorted(closes)
    if len(days) < _MIN_DAILY_RETURNS + 1:
        return empty
    returns = [
        abs(closes[days[index]] / closes[days[index - 1]] - 1)
        for index in range(1, len(days))
    ]
    typical = median(returns[-20:])
    if typical <= 0:
        return empty
    prior = None
    today = now.date()
    older = [day for day in days if day < today]
    if older:
        prior = closes[older[-1]]
    ref = reference_price if reference_price is not None else prior
    if ref is None or ref <= 0:
        return empty
    move = last / ref - 1
    return TapeReaction(
        triggered=abs(move) >= _REACTION_MULTIPLE * typical,
        last_price=last,
        reference_price=ref,
        move=move,
        typical_daily_move=typical,
    )


def visible_insights(
    insights: Sequence[PromotedInsight], *, symbol: str, now: datetime
) -> tuple[PromotedInsight, ...]:
    now = _aware(now, "now")
    return tuple(
        item
        for item in insights
        if item.symbol == symbol
        and item.knowledge_time <= now
        and (item.valid_until is None or item.valid_until >= now)
    )


PACKET_LOOKBACK = timedelta(days=2)


def recent_as_of(
    items: Sequence[object],
    now: datetime,
    lookback: timedelta = PACKET_LOOKBACK,
) -> tuple[object, ...]:
    now = _aware(now, "now")
    if lookback <= timedelta(0):
        raise ValueError("lookback must be positive")
    start = now - lookback
    found: list[object] = []
    for item in items:
        known = getattr(item, "knowledge_time", None)
        if known is None or known > now or known < start:
            continue
        found.append(item)
    return tuple(found)


def packet_news(
    items: Sequence[object],
    now: datetime,
    lookback: timedelta = PACKET_LOOKBACK,
    *,
    limit: int = 2,
) -> tuple[object, ...]:
    recent = recent_as_of(items, now, lookback)
    if recent:
        return recent
    ranked = [
        item
        for item in items
        if getattr(item, "knowledge_time", None) is not None
        and item.knowledge_time <= now
    ]
    ranked.sort(key=lambda item: item.knowledge_time, reverse=True)
    return tuple(ranked[: max(limit, 0)])


def visible_articles(
    articles: Sequence[Evidence], *, symbol: str, now: datetime, lookback: timedelta
) -> tuple[Evidence, ...]:
    now = _aware(now, "now")
    start = now - lookback
    found: list[Evidence] = []
    for article in articles:
        known = article.knowledge_time
        if known is None or known > now or known < start:
            continue
        if symbol not in article.symbols:
            continue
        found.append(article)
    return tuple(found)


def apply_trade_risk(
    candidate: TradeCandidate,
    *,
    action: TimingAction,
    size: float,
    facts: TapeFacts,
    reasons: Sequence[str],
    max_position_pct: float,
) -> TimingRiskDecision:
    if not isinstance(candidate, TradeCandidate):
        raise TypeError("candidate must be TradeCandidate")
    if not isinstance(action, TimingAction):
        raise TypeError("action must be TimingAction")
    if (
        isinstance(max_position_pct, bool)
        or not isinstance(max_position_pct, (int, float))
        or not isfinite(max_position_pct)
        or max_position_pct < 0
    ):
        raise ValueError("max_position_pct must be a non-negative number")
    if (
        isinstance(size, bool)
        or not isinstance(size, (int, float))
        or not isfinite(size)
        or size < 0
    ):
        raise ValueError("size must be a finite non-negative number")
    if candidate.direction is TradeSide.NO_TRADE:
        return TimingRiskDecision(
            TimingAction.REDUCE,
            replace(candidate, status=TradeCandidateStatus.REJECTED),
            facts,
            ("no_trade",),
        )
    capped = min(size, candidate.proposed_size, max_position_pct)
    notes = tuple(item for item in reasons if isinstance(item, str) and item.strip())
    if action is TimingAction.ALLOW and capped > 0:
        return TimingRiskDecision(
            TimingAction.ALLOW,
            replace(candidate, proposed_size=capped, status=TradeCandidateStatus.APPROVED),
            facts,
            notes or ("allow",),
        )
    if action is TimingAction.DEFER:
        return TimingRiskDecision(
            TimingAction.DEFER,
            replace(candidate, status=TradeCandidateStatus.REJECTED),
            facts,
            notes or ("defer",),
        )
    return TimingRiskDecision(
        TimingAction.REDUCE,
        replace(candidate, status=TradeCandidateStatus.REJECTED),
        facts,
        notes or ("reduce",),
    )


def pass_through_timing(
    candidate: TradeCandidate,
    *,
    tape: MarketTape,
    now: datetime,
    max_position_pct: float,
) -> TimingRiskDecision:
    facts = tape_facts(tape, candidate.instrument, now)
    if candidate.direction is TradeSide.NO_TRADE:
        return apply_trade_risk(
            candidate,
            action=TimingAction.REDUCE,
            size=0,
            facts=facts,
            reasons=("no_trade",),
            max_position_pct=max_position_pct,
        )
    return apply_trade_risk(
        candidate,
        action=TimingAction.ALLOW,
        size=candidate.proposed_size,
        facts=facts,
        reasons=("no_risk_agent",),
        max_position_pct=max_position_pct,
    )
