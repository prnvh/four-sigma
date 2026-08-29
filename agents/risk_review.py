from __future__ import annotations

from dataclasses import dataclass

from memory.position_risk import PositionRiskDecision, RiskCheckResult
from memory.types import RiskAnalysis


@dataclass(frozen=True, slots=True)
class FinalRiskReview:
    """AI commentary paired with the immutable deterministic trade decision."""

    deterministic: PositionRiskDecision
    advisory: RiskAnalysis

    def __post_init__(self) -> None:
        if not isinstance(self.deterministic, PositionRiskDecision):
            raise TypeError("deterministic must be PositionRiskDecision")
        if not isinstance(self.advisory, RiskAnalysis):
            raise TypeError("advisory must be RiskAnalysis")

    @property
    def result(self) -> RiskCheckResult:
        return self.deterministic.result

    @property
    def approved_size(self) -> float:
        return self.deterministic.approved_size


def finalize_risk_review(
    deterministic: PositionRiskDecision, advisory: RiskAnalysis
) -> FinalRiskReview:
    """Combine reviews without granting the AI any decision authority."""
    return FinalRiskReview(deterministic=deterministic, advisory=advisory)
