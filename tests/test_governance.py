import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memory import (
    AgentId,
    AuditEventType,
    AuditLedger,
    CanonicalId,
    CreatedAt,
    EntityId,
    GovernanceGate,
    GovernanceOutcome,
    PromotionProposal,
    ProposalId,
    ProposePermissions,
    SimulationClock,
    SimulationTime,
)


UTC = timezone.utc
START = SimulationTime(datetime(2024, 1, 1, tzinfo=UTC))
AAPL = EntityId("AAPL")
ANALYST = AgentId("news_analyst")


class _Ref(CanonicalId):
    __slots__ = ()


@dataclass(frozen=True)
class _EvidenceItem:
    created_at: CreatedAt
    expires_at: SimulationTime | None = None

    def visible_as_of(self, when: SimulationTime) -> bool:
        if self.created_at > when:
            return False
        if self.expires_at is not None and self.expires_at <= when:
            return False
        return True


class _Evidence:
    def __init__(self) -> None:
        self._items: dict[tuple[AgentId, CanonicalId], _EvidenceItem] = {}

    def add(
        self,
        agent_id: AgentId,
        ref: CanonicalId,
        created_at: CreatedAt,
        expires_at: SimulationTime | None = None,
    ) -> CanonicalId:
        self._items[(agent_id, ref)] = _EvidenceItem(created_at, expires_at)
        return ref

    def resolve(
        self,
        agent_id: AgentId,
        ref: CanonicalId,
        *,
        simulation_time: SimulationTime,
    ) -> _EvidenceItem | None:
        return self._items.get((agent_id, ref))


class _Shared:
    def __init__(self) -> None:
        self.writes: list[tuple[ProposalId, str, str, object]] = []

    def apply_approved(
        self, proposal: PromotionProposal, *, decided_at: SimulationTime
    ) -> None:
        self.writes.append(
            (
                proposal.id,
                proposal.target_resource,
                proposal.target_field,
                proposal.proposed_value,
            )
        )


def _schema_check(resource: str, field: str, value: object) -> str | None:
    allowed = {("insights", "claim"), ("insights", "direction")}
    if (resource, field) not in allowed:
        return f"unknown_field:{resource}.{field}"
    return None


class GovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = SimulationClock(START)
        self.evidence = _Evidence()
        self.shared = _Shared()
        self.ledger = AuditLedger()
        self.gate = GovernanceGate(
            permissions=ProposePermissions(
                {ANALYST: frozenset({("insights", "claim")})}
            ),
            evidence=self.evidence,
            schema_check=_schema_check,
            shared=self.shared,
            ledger=self.ledger,
        )

    def _created(self) -> CreatedAt:
        return CreatedAt(self.clock.now().value)

    def _ref(self, name: str = "ev-1") -> CanonicalId:
        ref = _Ref(name)
        return self.evidence.add(ANALYST, ref, self._created())

    def _proposal(
        self,
        *,
        proposal_id: str = "p1",
        evidence_refs: tuple[CanonicalId, ...] | None = None,
        resource: str = "insights",
        field: str = "claim",
        value: object = "demand is rising",
        confidence: float = 0.7,
        created_at: CreatedAt | None = None,
        agent_id: AgentId = ANALYST,
    ) -> PromotionProposal:
        refs = evidence_refs if evidence_refs is not None else (self._ref(),)
        return PromotionProposal(
            id=ProposalId(proposal_id),
            agent_id=agent_id,
            target_resource=resource,
            target_field=field,
            entity_id=AAPL,
            proposed_value=value,
            evidence_refs=refs,
            confidence=confidence,
            reasoning_summary="Synthetic test reasoning",
            created_at=created_at if created_at is not None else self._created(),
        )

    def test_approved_proposal_is_the_only_shared_write(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(), simulation_time=self.clock.now()
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.APPROVED)
        self.assertEqual(
            self.shared.writes,
            [(ProposalId("p1"), "insights", "claim", "demand is rising")],
        )
        types = [event.event_type for event in self.ledger.snapshot()]
        self.assertIn(AuditEventType.PROMOTION_APPROVED, types)
        self.assertIn(AuditEventType.SHARED_MEMORY_UPDATED, types)

    def test_rejected_proposal_does_not_write(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(field="direction", value="long"),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("permission_denied", decision.reasons)
        self.assertEqual(self.shared.writes, [])
        types = [event.event_type for event in self.ledger.snapshot()]
        self.assertIn(AuditEventType.PROMOTION_REJECTED, types)
        self.assertNotIn(AuditEventType.SHARED_MEMORY_UPDATED, types)

    def test_missing_evidence_rejected(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(evidence_refs=()),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("evidence_missing", decision.reasons)
        self.assertEqual(self.shared.writes, [])

    def test_other_agents_evidence_does_not_count(self) -> None:
        foreign = _Ref("foreign")
        self.evidence.add(AgentId("company_analyst"), foreign, self._created())
        decision = self.gate.evaluate(
            self._proposal(evidence_refs=(foreign,)),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("evidence_missing", decision.reasons)
        self.assertEqual(self.shared.writes, [])

    def test_future_evidence_rejected(self) -> None:
        later = SimulationTime(self.clock.now().value + timedelta(days=5))
        ref = _Ref("future")
        self.evidence.add(ANALYST, ref, CreatedAt(later.value))
        decision = self.gate.evaluate(
            self._proposal(evidence_refs=(ref,)),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("evidence_in_future", decision.reasons)
        self.assertEqual(self.shared.writes, [])

    def test_confidence_out_of_range_rejected(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(confidence=1.4),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("confidence_out_of_range", decision.reasons)
        self.assertEqual(self.shared.writes, [])

    def test_unknown_field_rejected(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(field="size", value=0.5),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertTrue(
            any(reason.startswith("unknown_field:") for reason in decision.reasons)
        )
        self.assertEqual(self.shared.writes, [])

    def test_duplicate_approved_proposal_rejected(self) -> None:
        refs = (self._ref(),)
        first = self.gate.evaluate(
            self._proposal(evidence_refs=refs),
            simulation_time=self.clock.now(),
        )
        second = self.gate.evaluate(
            self._proposal(proposal_id="p2", evidence_refs=refs),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(first.outcome, GovernanceOutcome.APPROVED)
        self.assertEqual(second.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("duplicate_proposal", second.reasons)
        self.assertEqual(len(self.shared.writes), 1)
        self.assertEqual(self.shared.writes[0][0], ProposalId("p1"))

    def test_future_proposal_is_deferred_without_write(self) -> None:
        later = CreatedAt(self.clock.now().value + timedelta(days=1))
        decision = self.gate.evaluate(
            self._proposal(created_at=later),
            simulation_time=self.clock.now(),
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.DEFERRED)
        self.assertEqual(decision.reasons, ("proposal_not_yet_created",))
        self.assertEqual(self.shared.writes, [])

    def test_same_inputs_yield_the_same_decision(self) -> None:
        refs = (self._ref("shared-ev"),)
        created = self._created()
        first = GovernanceGate(
            permissions=ProposePermissions(
                {ANALYST: frozenset({("insights", "claim")})}
            ),
            evidence=self.evidence,
            schema_check=_schema_check,
            shared=_Shared(),
            ledger=AuditLedger(),
        ).evaluate(
            self._proposal(proposal_id="a", evidence_refs=refs, created_at=created),
            simulation_time=START,
        )
        second = GovernanceGate(
            permissions=ProposePermissions(
                {ANALYST: frozenset({("insights", "claim")})}
            ),
            evidence=self.evidence,
            schema_check=_schema_check,
            shared=_Shared(),
            ledger=AuditLedger(),
        ).evaluate(
            self._proposal(proposal_id="b", evidence_refs=refs, created_at=created),
            simulation_time=START,
        )
        self.assertEqual(first.outcome, second.outcome)
        self.assertEqual(first.reasons, second.reasons)
        self.assertEqual(first.decided_at, second.decided_at)
