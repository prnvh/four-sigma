from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from memory.capabilities import Action, CAPABILITIES
from memory.portfolio import PortfolioSnapshot
from memory.portfolio_risk import PortfolioRiskComparison
from memory.position_risk import PositionRiskDecision, RiskCheckResult
from memory.types import PromotedInsight, TradeCandidate, jsonable

from .model import ModelClient
from .registry import PORTFOLIO_RISK_V1, AgentSpec


class PortfolioRiskRecommendation(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    RESIZE = "resize"
    DEFER = "defer"


PORTFOLIO_RISK_RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": [item.value for item in PortfolioRiskRecommendation],
        },
        "recommended_size": {"type": "number", "minimum": 0},
        "rationale": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "insight_refs": {
            "type": "array", "items": {"type": "string"}, "minItems": 1,
        },
    },
    "required": [
        "recommendation", "recommended_size", "rationale", "risk_flags",
        "insight_refs",
    ],
}


@dataclass(frozen=True, slots=True)
class PortfolioRiskAgentContext:
    candidate: TradeCandidate
    portfolio: PortfolioSnapshot
    deterministic: PositionRiskDecision
    comparison: PortfolioRiskComparison
    insight_summaries: tuple[PromotedInsight, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TradeCandidate):
            raise TypeError("candidate must be TradeCandidate")
        if not isinstance(self.portfolio, PortfolioSnapshot):
            raise TypeError("portfolio must be PortfolioSnapshot")
        if not isinstance(self.deterministic, PositionRiskDecision):
            raise TypeError("deterministic must be PositionRiskDecision")
        if not isinstance(self.comparison, PortfolioRiskComparison):
            raise TypeError("comparison must be PortfolioRiskComparison")
        if self.comparison.trade_id != self.candidate.id.value:
            raise ValueError("risk comparison must belong to the proposed trade")
        if self.comparison.before.knowledge_time != self.portfolio.knowledge_time:
            raise ValueError("portfolio and risk comparison must share knowledge_time")
        if not self.insight_summaries:
            raise ValueError("portfolio risk context requires selected insight summaries")
        if any(not isinstance(item, PromotedInsight) for item in self.insight_summaries):
            raise TypeError("insight_summaries must contain PromotedInsight values")
        if any(item.symbol != self.candidate.instrument for item in self.insight_summaries):
            raise ValueError("selected insights must belong to the proposed instrument")
        if any(item.knowledge_time > self.candidate.knowledge_time for item in self.insight_summaries):
            raise ValueError("selected insights cannot come from the future")
        if any(
            item.valid_until is not None and item.valid_until < self.candidate.knowledge_time
            for item in self.insight_summaries
        ):
            raise ValueError("selected insights cannot be expired")
        selected = {item.ref for item in self.insight_summaries}
        causal = {item.value for item in self.candidate.thesis_refs}
        if selected != causal:
            raise ValueError("selected insights must match the trade's thesis references")


@dataclass(frozen=True, slots=True)
class PortfolioRiskAssessment:
    recommendation: PortfolioRiskRecommendation
    recommended_size: float
    final_recommendation: PortfolioRiskRecommendation
    final_size: float
    rationale: str
    risk_flags: tuple[str, ...]
    insight_refs: tuple[str, ...]
    deterministic_result: RiskCheckResult


class PortfolioRiskAgent:
    """Advisory portfolio-level review with deterministic limits as the ceiling."""

    def __init__(
        self, model: ModelClient, spec: AgentSpec = PORTFOLIO_RISK_V1
    ) -> None:
        if spec.name != "portfolio_risk":
            raise ValueError("PortfolioRiskAgent requires a portfolio_risk spec")
        self.model = model
        self.spec = spec

    def analyze(self, context: PortfolioRiskAgentContext) -> PortfolioRiskAssessment:
        if not isinstance(context, PortfolioRiskAgentContext):
            raise TypeError("PortfolioRiskAgent requires PortfolioRiskAgentContext")
        CAPABILITIES.require("portfolio_risk", Action.READ, "portfolio", "positions")
        CAPABILITIES.require("portfolio_risk", Action.READ, "trade_candidates", "*")
        CAPABILITIES.require("portfolio_risk", Action.READ, "risk", "*")
        allowed_refs = {item.ref for item in context.insight_summaries}
        result = self.model.generate_json(
            instructions=self.spec.prompt,
            input_data={
                "proposed_trade": jsonable(context.candidate),
                "portfolio": jsonable(context.portfolio),
                "deterministic_risk": {
                    "result": context.deterministic.result.value,
                    "requested_size": context.deterministic.requested_size,
                    "approved_size": context.deterministic.approved_size,
                    "reasons": [item.as_dict() for item in context.deterministic.reasons],
                },
                "portfolio_risk_before": jsonable(context.comparison.before),
                "portfolio_risk_after": jsonable(context.comparison.after),
                "selected_insight_summaries": [
                    {
                        "ref": item.ref,
                        "claim": item.claim,
                        "direction": item.direction.value,
                        "confidence": item.confidence,
                    }
                    for item in context.insight_summaries
                ],
            },
            schema=PORTFOLIO_RISK_RECOMMENDATION_SCHEMA,
        )
        recommendation, size, rationale, flags, refs = self._validate(
            result, context.candidate.proposed_size, allowed_refs
        )
        final_recommendation, final_size = _apply_deterministic_ceiling(
            recommendation, size, context.deterministic
        )
        return PortfolioRiskAssessment(
            recommendation=recommendation,
            recommended_size=size,
            final_recommendation=final_recommendation,
            final_size=final_size,
            rationale=rationale,
            risk_flags=flags,
            insight_refs=refs,
            deterministic_result=context.deterministic.result,
        )

    @staticmethod
    def _validate(result: object, proposed_size: float, allowed_refs: set[str]):
        required = set(PORTFOLIO_RISK_RECOMMENDATION_SCHEMA["required"])
        if not isinstance(result, dict) or set(result) != required:
            raise ValueError("portfolio risk output has missing or unexpected fields")
        try:
            recommendation = PortfolioRiskRecommendation(result["recommendation"])
        except (TypeError, ValueError) as error:
            raise ValueError("portfolio risk recommendation is invalid") from error
        size = result["recommended_size"]
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not isfinite(size):
            raise ValueError("recommended_size must be a finite number")
        size = float(size)
        if not 0 <= size <= proposed_size:
            raise ValueError("recommended_size cannot exceed the proposed size")
        if recommendation in {
            PortfolioRiskRecommendation.REJECT, PortfolioRiskRecommendation.DEFER
        } and size != 0:
            raise ValueError("reject and defer recommendations require size zero")
        if recommendation is PortfolioRiskRecommendation.APPROVE and size != proposed_size:
            raise ValueError("approve recommendation must retain the proposed size")
        if recommendation is PortfolioRiskRecommendation.RESIZE and not 0 < size < proposed_size:
            raise ValueError("resize recommendation must reduce to a positive size")
        rationale = result["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be non-empty")
        for field in ("risk_flags", "insight_refs"):
            if not isinstance(result[field], list) or not all(
                isinstance(item, str) and item.strip() for item in result[field]
            ):
                raise ValueError(f"{field} must contain non-empty strings")
        refs = tuple(dict.fromkeys(result["insight_refs"]))
        if not refs or set(refs) - allowed_refs:
            raise ValueError("portfolio risk output cited unknown insights")
        flags = tuple(dict.fromkeys(item.strip() for item in result["risk_flags"]))
        return recommendation, size, rationale.strip(), flags, refs


def _apply_deterministic_ceiling(
    recommendation: PortfolioRiskRecommendation,
    recommended_size: float,
    deterministic: PositionRiskDecision,
) -> tuple[PortfolioRiskRecommendation, float]:
    if deterministic.result is RiskCheckResult.REJECT or deterministic.approved_size <= 0:
        return PortfolioRiskRecommendation.REJECT, 0.0
    if recommendation is PortfolioRiskRecommendation.REJECT:
        return recommendation, 0.0
    if recommendation is PortfolioRiskRecommendation.DEFER:
        return recommendation, 0.0
    final_size = min(recommended_size, deterministic.approved_size)
    if final_size < deterministic.requested_size:
        return PortfolioRiskRecommendation.RESIZE, final_size
    return PortfolioRiskRecommendation.APPROVE, final_size
