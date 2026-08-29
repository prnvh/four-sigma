from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .governance_gate import Proposal, RuleEvaluation
from .types import Direction, Finding, InsightRevision, SimulationTime


class NewsEvidenceResolver(Protocol):
    def resolve(self, agent_id, ref, *, simulation_time: SimulationTime): ...


@dataclass(frozen=True, slots=True)
class _ApprovedNewsInsight:
    entity: str
    claim_key: str
    direction: Direction


class NewsInsightGovernanceRules:
    """Deterministic quality rules applied only to News Analyst insights."""

    def __init__(
        self,
        *,
        evidence: NewsEvidenceResolver,
        allowed_source_classes: Iterable[str],
        min_evidence_count: int = 2,
    ) -> None:
        if (
            isinstance(min_evidence_count, bool)
            or not isinstance(min_evidence_count, int)
            or min_evidence_count < 1
        ):
            raise ValueError("min_evidence_count must be a positive integer")
        classes = frozenset(_key(item) for item in allowed_source_classes)
        if not classes or "" in classes:
            raise ValueError("allowed_source_classes must contain non-empty strings")
        self._evidence = evidence
        self._allowed_source_classes = classes
        self._min_evidence_count = min_evidence_count
        self._approved: list[_ApprovedNewsInsight] = []

    def evaluate(
        self, proposal: Proposal, *, simulation_time: SimulationTime
    ) -> RuleEvaluation:
        if proposal.agent_id.value != "news_analyst" or proposal.target_resource != "insights":
            return RuleEvaluation()

        reasons: list[str] = []
        tags: list[str] = []
        finding, revision = _news_finding(proposal.proposed_value)
        if finding is None or revision is None:
            return RuleEvaluation(reasons=("invalid_news_insight_schema",))
        if finding.subject.strip().upper() != proposal.entity_id.value.strip().upper():
            reasons.append("news_insight_entity_mismatch")
        if finding.confidence != proposal.confidence:
            reasons.append("news_confidence_mismatch")

        if revision.valid_until is None:
            reasons.append("news_insight_expiry_required")
        elif revision.valid_until.value <= simulation_time.value:
            reasons.append("news_insight_expired")

        unique_refs = tuple(dict.fromkeys(proposal.evidence_refs))
        if len(unique_refs) < self._min_evidence_count:
            reasons.append("insufficient_news_evidence")
        if {ref.value for ref in unique_refs} != set(finding.evidence_refs):
            reasons.append("news_evidence_mismatch")

        for ref in unique_refs:
            evidence = self._evidence.resolve(
                proposal.agent_id, ref, simulation_time=simulation_time
            )
            if evidence is None:
                continue  # The generic gate reports evidence_missing.
            source_class = getattr(evidence, "source_class", None)
            if not isinstance(source_class, str) or not source_class.strip():
                reasons.append("news_source_class_missing")
            elif _key(source_class) not in self._allowed_source_classes:
                reasons.append("news_source_class_not_allowed")

        claim_key = _claim_key(finding.claim)
        relevant = [item for item in self._approved if item.entity == proposal.entity_id.value]
        if any(item.claim_key == claim_key for item in relevant):
            reasons.append("duplicate_news_insight")
        if finding.direction is not Direction.NEUTRAL and any(
            item.direction is not Direction.NEUTRAL and item.direction is not finding.direction
            for item in relevant
        ):
            tags.append("conflicting_news_direction")

        return RuleEvaluation(
            reasons=tuple(dict.fromkeys(reasons)),
            tags=tuple(dict.fromkeys(tags)),
        )

    def record_approved(self, proposal: Proposal) -> None:
        if proposal.agent_id.value != "news_analyst" or proposal.target_resource != "insights":
            return
        finding, _ = _news_finding(proposal.proposed_value)
        if finding is not None:
            self._approved.append(
                _ApprovedNewsInsight(
                    entity=proposal.entity_id.value,
                    claim_key=_claim_key(finding.claim),
                    direction=finding.direction,
                )
            )


def _news_finding(value: object) -> tuple[Finding | None, InsightRevision | None]:
    if not isinstance(value, InsightRevision) or not isinstance(value.value, Finding):
        return None, None
    finding = value.value
    if finding.agent != "news_analyst":
        return None, None
    return finding, value


def _key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _claim_key(claim: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
