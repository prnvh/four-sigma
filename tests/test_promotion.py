import unittest
from datetime import datetime, timezone

from memory import (
    AgentId,
    AuditEventId,
    AuditLedger,
    CreatedAt,
    EntityId,
    EventId,
    PromotionProposal,
    ProposalId,
    SharedMemory,
    SharedMemorySection,
    WorkingMemory,
    WorkingMemoryCategory,
)


UTC = timezone.utc
CREATED_AT = CreatedAt(datetime(2025, 1, 1, tzinfo=UTC))
ANALYST = AgentId("news_analyst")
AAPL = EntityId("AAPL")


class PromotionProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = AuditLedger()
        self.working = WorkingMemory(self.ledger)
        self.shared = SharedMemory(self.ledger)
        self.audit_number = 0
        self.candidate = self._entry()

    def _entry(
        self,
        *,
        category: WorkingMemoryCategory = WorkingMemoryCategory.CANDIDATE_INSIGHT,
        value: object = None,
    ):
        self.audit_number += 1
        proposed = value if value is not None else {"claim": "demand is rising"}
        return self.working.for_agent(ANALYST).write(
            audit_event_id=AuditEventId(f"working-{self.audit_number}"),
            entity_id=AAPL,
            category=category,
            value=proposed,
            created_at=CREATED_AT,
        )

    def _proposal(self, entry=None, **overrides):
        values = {
            "id": ProposalId("proposal-1"),
            "agent_id": ANALYST,
            "target_resource": "insights",
            "target_field": "claim",
            "evidence_refs": (EventId("event-1"),),
            "confidence": 0.7,
            "reasoning_summary": "Supported by the cited event.",
            "created_at": CREATED_AT,
        }
        values.update(overrides)
        return PromotionProposal.from_working_memory(
            entry if entry is not None else self.candidate,
            **values,
        )

    def test_candidate_insight_becomes_proposal_without_shared_write(self) -> None:
        proposal = self._proposal()

        self.assertEqual(proposal.id, ProposalId("proposal-1"))
        self.assertEqual(proposal.agent_id, ANALYST)
        self.assertEqual(proposal.entity_id, AAPL)
        self.assertEqual(proposal.proposed_value, {"claim": "demand is rising"})
        for section in SharedMemorySection:
            self.assertEqual(dict(self.shared.snapshot(section)), {})

    def test_other_working_memory_categories_cannot_be_promoted(self) -> None:
        entry = self._entry(category=WorkingMemoryCategory.OBSERVATION)

        with self.assertRaises(ValueError):
            self._proposal(entry)

    def test_agent_cannot_promote_another_agents_entry(self) -> None:
        with self.assertRaises(PermissionError):
            self._proposal(agent_id=AgentId("company_analyst"))

    def test_proposed_value_is_detached_from_working_memory_value(self) -> None:
        value = {"reasons": ["demand"]}
        proposal = self._proposal(self._entry(value=value))
        value["reasons"].append("pricing")

        self.assertEqual(proposal.proposed_value, {"reasons": ["demand"]})

    def test_structural_fields_are_validated(self) -> None:
        cases = (
            ({"target_resource": " "}, ValueError),
            ({"target_field": ""}, ValueError),
            ({"reasoning_summary": ""}, ValueError),
            ({"confidence": True}, TypeError),
            ({"evidence_refs": ("event-1",)}, TypeError),
        )
        for overrides, error in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(error):
                    self._proposal(**overrides)

    def test_governance_rules_are_not_duplicated_in_proposal(self) -> None:
        proposal = self._proposal(
            target_resource="unknown_resource",
            target_field="unknown_field",
            evidence_refs=(),
            confidence=1.4,
        )

        self.assertEqual(proposal.target_resource, "unknown_resource")
        self.assertEqual(proposal.evidence_refs, ())
        self.assertEqual(proposal.confidence, 1.4)


if __name__ == "__main__":
    unittest.main()
