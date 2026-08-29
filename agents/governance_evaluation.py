from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from memory.audit_logger import AuditEvent, AuditEventType, AuditLedger
from memory.types import ProposalId, SimulationTime


class GovernanceEvaluationError(ValueError):
    """Governance audit and outcome data cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class GovernanceInsightOutcome:
    """Fixed-horizon, direction-adjusted performance for one proposal."""

    proposal_id: ProposalId
    performance: float
    evaluated_at: SimulationTime
    horizon: timedelta
    polluted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_id, ProposalId):
            raise TypeError("proposal_id must be ProposalId")
        if isinstance(self.performance, bool) or not isinstance(
            self.performance, (int, float)
        ):
            raise GovernanceEvaluationError("performance must be numeric")
        performance = float(self.performance)
        if not isfinite(performance):
            raise GovernanceEvaluationError("performance must be finite")
        if not isinstance(self.evaluated_at, SimulationTime):
            raise TypeError("evaluated_at must be SimulationTime")
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise GovernanceEvaluationError("horizon must be a positive timedelta")
        if not isinstance(self.polluted, bool):
            raise TypeError("polluted must be bool")
        object.__setattr__(self, "performance", performance)


@dataclass(frozen=True, slots=True)
class GovernanceEvaluation:
    total_proposals: int
    resolved_proposals: int
    unresolved_proposals: int
    approved_proposals: int
    rejected_proposals: int
    approval_rate: float | None
    rejection_rate: float | None
    approved_insight_performance: float | None
    rejected_insight_counterfactual_performance: float | None
    memory_pollution_rate: float | None
    duplicate_rate: float | None
    gate_value_added: float | None


class GovernanceEvaluator:
    """Evaluate promotion decisions without changing governance or shared memory."""

    def evaluate(
        self,
        audit: AuditLedger | Sequence[AuditEvent],
        *,
        outcomes: Sequence[GovernanceInsightOutcome] = (),
    ) -> GovernanceEvaluation:
        events = audit.snapshot() if isinstance(audit, AuditLedger) else tuple(audit)
        if any(not isinstance(event, AuditEvent) for event in events):
            raise TypeError("audit must contain AuditEvent values")

        decisions: dict[ProposalId, AuditEvent] = {}
        for event in events:
            if event.event_type not in {
                AuditEventType.PROMOTION_APPROVED,
                AuditEventType.PROMOTION_REJECTED,
            }:
                continue
            if not isinstance(event.subject_id, ProposalId):
                raise GovernanceEvaluationError(
                    "promotion decision requires a ProposalId subject"
                )
            if event.subject_id in decisions:
                raise GovernanceEvaluationError(
                    f"duplicate decision for proposal: {event.subject_id}"
                )
            decisions[event.subject_id] = event

        outcome_by_proposal: dict[ProposalId, GovernanceInsightOutcome] = {}
        horizons: set[timedelta] = set()
        for outcome in outcomes:
            if not isinstance(outcome, GovernanceInsightOutcome):
                raise TypeError("outcomes must contain GovernanceInsightOutcome values")
            if outcome.proposal_id in outcome_by_proposal:
                raise GovernanceEvaluationError(
                    f"duplicate outcome for proposal: {outcome.proposal_id}"
                )
            outcome_by_proposal[outcome.proposal_id] = outcome
            horizons.add(outcome.horizon)
        if len(horizons) > 1:
            raise GovernanceEvaluationError("all outcomes must use the same horizon")

        unknown = set(outcome_by_proposal) - set(decisions)
        if unknown:
            raise GovernanceEvaluationError(
                "outcomes reference unknown proposals: "
                + ", ".join(sorted(item.value for item in unknown))
            )
        for proposal_id, outcome in outcome_by_proposal.items():
            decision = decisions[proposal_id]
            expected = decision.occurred_at.value + outcome.horizon
            if outcome.evaluated_at.value != expected:
                raise GovernanceEvaluationError(
                    f"outcome for {proposal_id.value} is not at its declared horizon"
                )

        approved = {
            proposal_id
            for proposal_id, event in decisions.items()
            if event.event_type is AuditEventType.PROMOTION_APPROVED
        }
        rejected = set(decisions) - approved
        approved_performance = [
            outcome_by_proposal[item].performance
            for item in approved & set(outcome_by_proposal)
        ]
        rejected_performance = [
            outcome_by_proposal[item].performance
            for item in rejected & set(outcome_by_proposal)
        ]
        approved_average = _average(approved_performance)
        rejected_average = _average(rejected_performance)
        resolved_approved = approved & set(outcome_by_proposal)
        polluted = sum(outcome_by_proposal[item].polluted for item in resolved_approved)
        duplicates = sum(_is_duplicate(event) for event in decisions.values())
        total = len(decisions)

        return GovernanceEvaluation(
            total_proposals=total,
            resolved_proposals=len(outcome_by_proposal),
            unresolved_proposals=total - len(outcome_by_proposal),
            approved_proposals=len(approved),
            rejected_proposals=len(rejected),
            approval_rate=len(approved) / total if total else None,
            rejection_rate=len(rejected) / total if total else None,
            approved_insight_performance=approved_average,
            rejected_insight_counterfactual_performance=rejected_average,
            memory_pollution_rate=(
                polluted / len(resolved_approved) if resolved_approved else None
            ),
            duplicate_rate=duplicates / total if total else None,
            gate_value_added=(
                approved_average - rejected_average
                if approved_average is not None and rejected_average is not None
                else None
            ),
        )


def _average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_duplicate(event: AuditEvent) -> bool:
    reasons = event.details.get("reasons", ())
    if not isinstance(reasons, tuple):
        raise GovernanceEvaluationError("promotion reasons must be a sequence")
    if any(not isinstance(reason, str) for reason in reasons):
        raise GovernanceEvaluationError("promotion reasons must contain strings")
    return any("duplicate" in reason for reason in reasons)
