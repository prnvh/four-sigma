from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from memory.capabilities import CAPABILITIES
from memory.evidence_refs import bound_cited_refs
from memory.execution import MarketTape
from memory.timing_risk import (
    TimingAction,
    TimingRiskDecision,
    apply_tape_ceiling,
    apply_trend_ceiling,
    apply_trade_risk,
    pass_through_timing,
    tape_facts,
    tape_reaction,
    visible_articles,
    visible_insights,
)
from memory.types import Evidence, PromotedInsight, TradeCandidate, TradeSide, jsonable

from .model import ModelClient
from .registry import TRADE_RISK_V1, AgentSpec


TRADE_RISK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": [item.value for item in TimingAction]},
        "size": {"type": "number", "minimum": 0},
        "rationale": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["action", "size", "rationale", "evidence_refs"],
}


class TradeRiskAnalyst:
    """Judges whether this trade is well timed. Advisory. Defaults to allowing it."""

    def __init__(self, model: ModelClient, spec: AgentSpec = TRADE_RISK_V1) -> None:
        if spec.name != "risk_llm":
            raise ValueError("TradeRiskAnalyst requires a risk_llm spec")
        self.model = model
        self.spec = spec

    def review(
        self,
        candidate: TradeCandidate,
        *,
        tape: MarketTape,
        insights: Sequence[PromotedInsight],
        articles: Sequence[Evidence],
        now: datetime,
        existing_quantity: float = 0,
        max_position_pct: float = 1.0,
    ) -> TimingRiskDecision:
        if not isinstance(candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if candidate.direction is TradeSide.NO_TRADE:
            return pass_through_timing(
                candidate, tape=tape, now=now, max_position_pct=max_position_pct
            )
        CAPABILITIES.require_reads(
            self.spec.name, (("insights", "*"), ("events", "*"), ("market", "features"))
        )
        facts = tape_facts(tape, candidate.instrument, now)
        known_insights = visible_insights(
            insights, symbol=candidate.instrument, now=now
        )
        known_articles = visible_articles(
            articles,
            symbol=candidate.instrument,
            now=now,
            lookback=timedelta(days=2),
        )
        reaction = tape_reaction(tape, candidate.instrument, now)
        if not known_insights and not known_articles:
            return apply_trend_ceiling(
                apply_tape_ceiling(
                    apply_trade_risk(
                        candidate,
                        action=TimingAction.ALLOW,
                        size=candidate.proposed_size,
                        facts=facts,
                        reasons=("no_sourced_context",),
                        max_position_pct=max_position_pct,
                    ),
                    reaction,
                )
            )
        try:
            result = self.model.generate_json(
                instructions=self.spec.prompt,
                input_data={
                    "proposed_trade": jsonable(candidate),
                    "existing_quantity": existing_quantity,
                    "max_position_pct": max_position_pct,
                    "tape_facts": jsonable(facts),
                    "insights": [jsonable(item) for item in known_insights],
                    "articles": [jsonable(item) for item in known_articles],
                },
                schema=TRADE_RISK_SCHEMA,
            )
            action, size, reasons = _parse_result(
                result, candidate, (*known_insights, *known_articles)
            )
        except (ValueError, RuntimeError, TypeError) as exc:
            return apply_trend_ceiling(
                apply_tape_ceiling(
                    apply_trade_risk(
                        candidate,
                        action=TimingAction.ALLOW,
                        size=candidate.proposed_size,
                        facts=facts,
                        reasons=(f"risk_pass_through:{exc}",),
                        max_position_pct=max_position_pct,
                    ),
                    reaction,
                )
            )
        return apply_trend_ceiling(
            apply_tape_ceiling(
                apply_trade_risk(
                    candidate,
                    action=action,
                    size=size,
                    facts=facts,
                    reasons=reasons,
                    max_position_pct=max_position_pct,
                ),
                reaction,
            )
        )


def _parse_result(
    result: object,
    candidate: TradeCandidate,
    evidence: Sequence[object],
) -> tuple[TimingAction, float, tuple[str, ...]]:
    if not isinstance(result, dict):
        raise ValueError("trade risk output must be an object")
    required = {"action", "size", "rationale", "evidence_refs"}
    if set(result) != required:
        raise ValueError("trade risk output has missing or unexpected fields")
    try:
        action = TimingAction(result["action"])
    except ValueError as exc:
        raise ValueError("action is invalid") from exc
    size = result["size"]
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        raise ValueError("size must be numeric")
    if size > candidate.proposed_size:
        raise ValueError("size cannot exceed the proposed trade")
    if not isinstance(result["rationale"], str) or not result["rationale"].strip():
        raise ValueError("rationale must be a non-empty string")
    if not isinstance(result["evidence_refs"], list) or not all(
        isinstance(ref, str) for ref in result["evidence_refs"]
    ):
        raise ValueError("evidence_refs must be a list of strings")
    cited = bound_cited_refs(result["evidence_refs"], evidence)
    if not cited:
        raise ValueError("trade risk cited no supplied evidence")
    return action, float(size), (result["rationale"].strip(),)
