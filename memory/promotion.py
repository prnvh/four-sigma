from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from .ids import AgentId, CanonicalId, EntityId, ProposalId
from .time import CreatedAt
from .working import WorkingMemoryCategory, WorkingMemoryEntry


@dataclass(frozen=True, slots=True, kw_only=True)
class PromotionProposal:
    """An agent request for governance to consider a shared-memory change."""

    id: ProposalId
    agent_id: AgentId
    target_resource: str
    target_field: str
    entity_id: EntityId
    proposed_value: object
    evidence_refs: tuple[CanonicalId, ...]
    confidence: float
    reasoning_summary: str
    created_at: CreatedAt

    def __post_init__(self) -> None:
        if not isinstance(self.id, ProposalId):
            raise TypeError("id must be ProposalId")
        if not isinstance(self.agent_id, AgentId):
            raise TypeError("agent_id must be AgentId")
        if not isinstance(self.entity_id, EntityId):
            raise TypeError("entity_id must be EntityId")
        if not isinstance(self.created_at, CreatedAt):
            raise TypeError("created_at must be CreatedAt")

        resource = _required_text(self.target_resource, "target_resource")
        field = _required_text(self.target_field, "target_field")
        reasoning = _required_text(self.reasoning_summary, "reasoning_summary")
        evidence = _evidence_refs(self.evidence_refs)
        confidence = _confidence(self.confidence)

        object.__setattr__(self, "target_resource", resource)
        object.__setattr__(self, "target_field", field)
        object.__setattr__(self, "reasoning_summary", reasoning)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "proposed_value", deepcopy(self.proposed_value))

    @classmethod
    def from_working_memory(
        cls,
        entry: WorkingMemoryEntry,
        *,
        id: ProposalId,
        agent_id: AgentId,
        target_resource: str,
        target_field: str,
        evidence_refs: Sequence[CanonicalId],
        confidence: float,
        reasoning_summary: str,
        created_at: CreatedAt,
    ) -> PromotionProposal:
        if not isinstance(entry, WorkingMemoryEntry):
            raise TypeError("entry must be WorkingMemoryEntry")
        if entry.category is not WorkingMemoryCategory.CANDIDATE_INSIGHT:
            raise ValueError("only candidate_insight can become a promotion proposal")
        if not isinstance(agent_id, AgentId):
            raise TypeError("agent_id must be AgentId")
        if entry.agent_id != agent_id:
            raise PermissionError("agent can only promote its own working memory")

        return cls(
            id=id,
            agent_id=agent_id,
            target_resource=target_resource,
            target_field=target_field,
            entity_id=entry.entity_id,
            proposed_value=entry.value,
            evidence_refs=tuple(evidence_refs),
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            created_at=created_at,
        )


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _evidence_refs(value: object) -> tuple[CanonicalId, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("evidence_refs must be a sequence of CanonicalId values")
    refs = tuple(value)
    if any(not isinstance(ref, CanonicalId) for ref in refs):
        raise TypeError("evidence_refs must contain only CanonicalId values")
    return refs


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("confidence must be numeric")
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError("confidence must be finite")
    return confidence
