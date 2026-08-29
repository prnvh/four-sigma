from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Protocol

from .audit_logger import AuditEventType, AuditLedger
from .shared_mem import SharedMemoryValidationError
from .types import AgentId, AuditEventId, CanonicalId, CreatedAt, EntityId, ProposalId, SimulationTime


class GovernanceOutcome(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"


@dataclass(frozen=True, slots=True, kw_only=True)
class GovernanceDecision:
    proposal_id: ProposalId
    outcome: GovernanceOutcome
    reasons: tuple[str, ...]
    decided_at: SimulationTime
    tags: tuple[str, ...] = ()


class Proposal(Protocol):
    id: ProposalId
    agent_id: AgentId
    target_resource: str
    target_field: str
    entity_id: EntityId
    proposed_value: object
    evidence_refs: Sequence[CanonicalId]
    confidence: float
    reasoning_summary: str
    created_at: CreatedAt


class Evidence(Protocol):
    created_at: CreatedAt

    def visible_as_of(self, when: SimulationTime) -> bool: ...


class EvidenceResolver(Protocol):
    def resolve(
        self,
        agent_id: AgentId,
        ref: CanonicalId,
        *,
        simulation_time: SimulationTime,
    ) -> Evidence | None: ...


class SharedWriter(Protocol):
    def apply_approved(
        self, proposal: Proposal, *, decided_at: SimulationTime
    ) -> None: ...


SchemaCheck = Callable[[str, str, object], str | None]


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    reasons: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


class GovernanceRule(Protocol):
    def evaluate(
        self, proposal: Proposal, *, simulation_time: SimulationTime
    ) -> RuleEvaluation: ...

    def record_approved(self, proposal: Proposal) -> None: ...


class ProposePermissions:
    """Deny-by-default map of agent → {(resource, field)} they may propose."""

    def __init__(
        self, allowed: Mapping[AgentId, frozenset[tuple[str, str]]] | None = None
    ) -> None:
        self._allowed = {
            agent_id: frozenset(pairs) for agent_id, pairs in (allowed or {}).items()
        }

    def allows(self, agent_id: AgentId, resource: str, field: str) -> bool:
        return (resource, field) in self._allowed.get(agent_id, frozenset())


class GovernanceGate:
    """Deterministic promotion gate. Shared writes happen only on APPROVED."""

    def __init__(
        self,
        *,
        permissions: ProposePermissions,
        evidence: EvidenceResolver,
        schema_check: SchemaCheck,
        shared: SharedWriter,
        ledger: AuditLedger,
        rules: Sequence[GovernanceRule] = (),
    ) -> None:
        self._permissions = permissions
        self._evidence = evidence
        self._schema_check = schema_check
        self._shared = shared
        self._ledger = ledger
        self._rules = tuple(rules)
        self._seen: set[tuple[object, ...]] = set()
        self._evaluated_ids: set[ProposalId] = set()
        self._evaluations = 0

    def evaluate(
        self, proposal: Proposal, *, simulation_time: SimulationTime
    ) -> GovernanceDecision:
        if not isinstance(simulation_time, SimulationTime):
            raise TypeError("simulation_time must be SimulationTime")
        _require_proposal(proposal)

        self._evaluations += 1
        occurred_at = CreatedAt(simulation_time.value)
        self._record(
            AuditEventType.PROMOTION_REQUESTED,
            proposal,
            occurred_at,
            {"evaluation": self._evaluations},
        )

        if proposal.created_at > simulation_time:
            return self._decide(
                proposal,
                simulation_time,
                occurred_at,
                GovernanceOutcome.DEFERRED,
                ("proposal_not_yet_created",),
            )

        reasons = list(self._check(proposal, simulation_time))
        tags: list[str] = []
        for rule in self._rules:
            result = rule.evaluate(proposal, simulation_time=simulation_time)
            if not isinstance(result, RuleEvaluation):
                raise TypeError("governance rules must return RuleEvaluation")
            reasons.extend(result.reasons)
            tags.extend(result.tags)
        if reasons:
            return self._decide(
                proposal,
                simulation_time,
                occurred_at,
                GovernanceOutcome.REJECTED,
                tuple(dict.fromkeys(reasons)),
                tuple(dict.fromkeys(tags)),
            )

        try:
            self._ledger.record_state_change(
                event_id=self._audit_id("shared", proposal),
                event_type=AuditEventType.SHARED_MEMORY_UPDATED,
                occurred_at=occurred_at,
                change=lambda: self._shared.apply_approved(
                    proposal, decided_at=simulation_time
                ),
                details={
                    "resource": proposal.target_resource,
                    "field": proposal.target_field,
                    "entity_id": proposal.entity_id.value,
                },
                agent_id=proposal.agent_id,
                subject_id=proposal.id,
            )
        except SharedMemoryValidationError as error:
            self._record(
                AuditEventType.SHARED_MEMORY_WRITE_REJECTED,
                proposal,
                occurred_at,
                {"reason": type(error).__name__, "message": str(error)},
            )
            return self._decide(
                proposal,
                simulation_time,
                occurred_at,
                GovernanceOutcome.REJECTED,
                ("shared_memory_rejected",),
            )
        self._seen.add(_fingerprint(proposal))
        for rule in self._rules:
            rule.record_approved(proposal)
        return self._decide(
            proposal,
            simulation_time,
            occurred_at,
            GovernanceOutcome.APPROVED,
            (),
            tuple(dict.fromkeys(tags)),
        )

    def _check(
        self, proposal: Proposal, simulation_time: SimulationTime
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self._permissions.allows(
            proposal.agent_id, proposal.target_resource, proposal.target_field
        ):
            reasons.append("permission_denied")
        violation = self._schema_check(
            proposal.target_resource,
            proposal.target_field,
            proposal.proposed_value,
        )
        if violation is not None:
            reasons.append(violation)
        if not _in_unit_interval(proposal.confidence):
            reasons.append("confidence_out_of_range")
        reasons.extend(self._evidence_reasons(proposal, simulation_time))
        if (
            proposal.id in self._evaluated_ids
            or _fingerprint(proposal) in self._seen
        ):
            reasons.append("duplicate_proposal")
        return tuple(reasons)

    def _evidence_reasons(
        self, proposal: Proposal, simulation_time: SimulationTime
    ) -> list[str]:
        if not proposal.evidence_refs:
            return ["evidence_missing"]
        reasons: list[str] = []
        for ref in proposal.evidence_refs:
            entry = self._evidence.resolve(
                proposal.agent_id, ref, simulation_time=simulation_time
            )
            if entry is None:
                reasons.append("evidence_missing")
                continue
            if entry.created_at > simulation_time:
                reasons.append("evidence_in_future")
                continue
            if not entry.visible_as_of(simulation_time):
                reasons.append("evidence_missing")
        return reasons

    def _decide(
        self,
        proposal: Proposal,
        simulation_time: SimulationTime,
        occurred_at: CreatedAt,
        outcome: GovernanceOutcome,
        reasons: tuple[str, ...],
        tags: tuple[str, ...] = (),
    ) -> GovernanceDecision:
        if outcome is not GovernanceOutcome.DEFERRED:
            self._evaluated_ids.add(proposal.id)
            self._record(
                (
                    AuditEventType.PROMOTION_APPROVED
                    if outcome is GovernanceOutcome.APPROVED
                    else AuditEventType.PROMOTION_REJECTED
                ),
                proposal,
                occurred_at,
                {
                    "outcome": outcome.value,
                    "reasons": list(reasons),
                    "tags": list(tags),
                },
            )
        return GovernanceDecision(
            proposal_id=proposal.id,
            outcome=outcome,
            reasons=reasons,
            decided_at=simulation_time,
            tags=tags,
        )

    def _record(
        self,
        event_type: AuditEventType,
        proposal: Proposal,
        occurred_at: CreatedAt,
        details: Mapping[str, object],
    ) -> None:
        self._ledger.append(
            event_id=self._audit_id(event_type.value, proposal),
            event_type=event_type,
            occurred_at=occurred_at,
            details=dict(details),
            agent_id=proposal.agent_id,
            subject_id=proposal.id,
        )

    def _audit_id(self, kind: str, proposal: Proposal) -> AuditEventId:
        return AuditEventId(f"{kind}:{self._evaluations}:{proposal.id.value}")


def _require_proposal(proposal: object) -> None:
    required = (
        "id",
        "agent_id",
        "target_resource",
        "target_field",
        "entity_id",
        "proposed_value",
        "evidence_refs",
        "confidence",
        "reasoning_summary",
        "created_at",
    )
    missing = [name for name in required if not hasattr(proposal, name)]
    if missing:
        raise TypeError(f"proposal is missing {', '.join(missing)}")


def _fingerprint(proposal: Proposal) -> tuple[object, ...]:
    return (
        proposal.agent_id,
        proposal.target_resource,
        proposal.target_field,
        proposal.entity_id,
        _freeze(proposal.proposed_value),
    )


def _freeze(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, _freeze(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _in_unit_interval(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return 0.0 <= float(value) <= 1.0
