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
    InsightId,
    InsightRevision,
    InsightStatus,
    PromotionProposal,
    ProposalId,
    ProposePermissions,
    SharedMemory,
    SimulationTime,
)


UTC = timezone.utc
T0 = SimulationTime(datetime(2025, 1, 1, tzinfo=UTC))
T1 = SimulationTime(T0.value + timedelta(days=1))
T2 = SimulationTime(T0.value + timedelta(days=2))
T3 = SimulationTime(T0.value + timedelta(days=3))
AGENT = AgentId("company_analyst")
ENTITY = EntityId("AAPL")
INSIGHT = InsightId("aapl-demand")


class _EvidenceId(CanonicalId):
    __slots__ = ()


@dataclass(frozen=True)
class _EvidenceItem:
    created_at: CreatedAt

    def visible_as_of(self, when: SimulationTime) -> bool:
        return self.created_at.value <= when.value


class _Evidence:
    def __init__(self) -> None:
        self.item = _EvidenceItem(CreatedAt(T0.value))

    def resolve(
        self,
        agent_id: AgentId,
        ref: CanonicalId,
        *,
        simulation_time: SimulationTime,
    ) -> _EvidenceItem:
        return self.item


class InsightVersioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AuditLedger()
        self.shared = SharedMemory(self.ledger)
        self.gate = GovernanceGate(
            permissions=ProposePermissions(
                {AGENT: frozenset({("insights", "claim")})}
            ),
            evidence=_Evidence(),
            schema_check=lambda resource, field, value: None,
            shared=self.shared,
            ledger=self.ledger,
        )

    def _proposal(
        self,
        proposal_id: str,
        value: object,
        *,
        created_at: SimulationTime = T0,
    ) -> PromotionProposal:
        return PromotionProposal(
            id=ProposalId(proposal_id),
            agent_id=AGENT,
            target_resource="insights",
            target_field="claim",
            entity_id=ENTITY,
            proposed_value=value,
            evidence_refs=(_EvidenceId("source"),),
            confidence=0.8,
            reasoning_summary="Versioning test",
            created_at=CreatedAt(created_at.value),
        )

    def _approve(
        self, proposal_id: str, revision: InsightRevision, when: SimulationTime
    ) -> None:
        decision = self.gate.evaluate(
            self._proposal(proposal_id, revision, created_at=when),
            simulation_time=when,
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.APPROVED)

    def test_new_version_preserves_old_version_and_links_to_it(self) -> None:
        first_value = {"claim": "demand is rising"}
        self._approve(
            "p1", InsightRevision(insight_id=INSIGHT, value=first_value), T0
        )
        first_value["claim"] = "caller changed it"
        self._approve(
            "p2", InsightRevision(insight_id=INSIGHT, value="demand is flat"), T1
        )

        history = self.shared.insight_history(INSIGHT)
        self.assertEqual([item.version for item in history], [1, 2])
        self.assertEqual([item.supersedes_version for item in history], [None, 1])
        self.assertEqual(history[0].value, {"claim": "demand is rising"})
        self.assertEqual(history[0].created_by_proposal, ProposalId("p1"))
        self.assertEqual(history[1].created_by_proposal, ProposalId("p2"))

    def test_historical_state_is_reconstructed_at_each_simulation_time(self) -> None:
        self._approve("p1", InsightRevision(insight_id=INSIGHT, value="rising"), T0)
        self._approve("p2", InsightRevision(insight_id=INSIGHT, value="flat"), T2)

        self.assertEqual(self.shared.insight_as_of(INSIGHT, T1).value, "rising")
        self.assertEqual(self.shared.insight_as_of(INSIGHT, T2).value, "flat")
        self.assertEqual(self.shared.insights_as_of(T1)[INSIGHT].version, 1)

    def test_same_time_uses_the_higher_deterministic_version(self) -> None:
        self._approve("p1", InsightRevision(insight_id=INSIGHT, value="first"), T1)
        self._approve("p2", InsightRevision(insight_id=INSIGHT, value="second"), T1)
        self.assertEqual(self.shared.insight_as_of(INSIGHT, T1).value, "second")

    def test_planned_expiry_removes_version_after_valid_until(self) -> None:
        self._approve(
            "p1",
            InsightRevision(insight_id=INSIGHT, value="temporary", valid_until=T2),
            T0,
        )
        self.assertIsNotNone(self.shared.insight_as_of(INSIGHT, T1))
        self.assertIsNone(self.shared.insight_as_of(INSIGHT, T2))

    def test_retraction_is_a_new_version_not_a_history_edit(self) -> None:
        self._approve("p1", InsightRevision(insight_id=INSIGHT, value="active"), T0)
        self._approve(
            "p2",
            InsightRevision(
                insight_id=INSIGHT,
                value="retracted",
                status=InsightStatus.RETRACTED,
            ),
            T2,
        )
        self.assertEqual(self.shared.insight_as_of(INSIGHT, T1).version, 1)
        self.assertIsNone(self.shared.insight_as_of(INSIGHT, T3))
        self.assertEqual(len(self.shared.insight_history(INSIGHT)), 2)

    def test_backdated_version_is_rejected_audited_and_not_stored(self) -> None:
        self._approve("p1", InsightRevision(insight_id=INSIGHT, value="current"), T2)
        decision = self.gate.evaluate(
            self._proposal(
                "p2",
                InsightRevision(insight_id=INSIGHT, value="backdated"),
                created_at=T1,
            ),
            simulation_time=T1,
        )

        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertEqual(decision.reasons, ("shared_memory_rejected",))
        self.assertEqual(len(self.shared.insight_history(INSIGHT)), 1)
        types = [event.event_type for event in self.ledger.snapshot()]
        self.assertIn(AuditEventType.SHARED_MEMORY_WRITE_REJECTED, types)
        self.assertEqual(types.count(AuditEventType.SHARED_MEMORY_UPDATED), 1)

    def test_plain_approved_insight_becomes_version_one(self) -> None:
        decision = self.gate.evaluate(
            self._proposal("plain-proposal", "raw working-memory insight"),
            simulation_time=T0,
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.APPROVED)
        history = self.shared.insight_history(InsightId("plain-proposal"))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].value, "raw working-memory insight")
        self.assertEqual(history[0].version, 1)

    def test_contradicting_insights_coexist_without_overwrite(self) -> None:
        first = InsightId("bull-case")
        second = InsightId("bear-case")
        self._approve("p1", InsightRevision(insight_id=first, value="bullish"), T0)
        self._approve(
            "p2",
            InsightRevision(
                insight_id=second,
                value="bearish",
                contradicts=(first,),
            ),
            T1,
        )

        visible = self.shared.insights_as_of(T1)
        self.assertEqual(set(visible), {first, second})
        self.assertEqual(visible[second].contradicts, (first,))
        self.assertEqual(len(self.shared.insight_history(first)), 1)

    def test_unknown_relationship_is_rejected_without_state_change(self) -> None:
        decision = self.gate.evaluate(
            self._proposal(
                "p1",
                InsightRevision(
                    insight_id=INSIGHT,
                    value="unsupported",
                    supports=(InsightId("missing"),),
                ),
            ),
            simulation_time=T0,
        )

        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertEqual(decision.reasons, ("shared_memory_rejected",))
        self.assertEqual(self.shared.insight_history(INSIGHT), ())


if __name__ == "__main__":
    unittest.main()
