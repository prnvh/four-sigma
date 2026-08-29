import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memory import (
    AgentId,
    AuditLedger,
    CanonicalId,
    CreatedAt,
    Direction,
    EntityId,
    Finding,
    GovernanceGate,
    GovernanceOutcome,
    InsightId,
    InsightRevision,
    NewsInsightGovernanceRules,
    PromotionProposal,
    ProposalId,
    ProposePermissions,
    SharedMemory,
    SimulationTime,
)


NOW = SimulationTime(datetime(2026, 8, 29, tzinfo=timezone.utc))
LATER = SimulationTime(NOW.value + timedelta(days=7))
AGENT = AgentId("news_analyst")
ENTITY = EntityId("ABC")


class _EvidenceId(CanonicalId):
    __slots__ = ()


@dataclass(frozen=True)
class _EvidenceItem:
    created_at: CreatedAt
    source_class: str | None

    def visible_as_of(self, when: SimulationTime) -> bool:
        return self.created_at.value <= when.value


class _EvidenceResolver:
    def __init__(self) -> None:
        self.items: dict[_EvidenceId, _EvidenceItem] = {}

    def add(self, name: str, source_class: str | None = "reputable_newswire") -> _EvidenceId:
        ref = _EvidenceId(name)
        self.items[ref] = _EvidenceItem(CreatedAt(NOW.value), source_class)
        return ref

    def resolve(self, agent_id, ref, *, simulation_time):
        return self.items.get(ref)


class NewsInsightGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = _EvidenceResolver()
        self.ledger = AuditLedger()
        self.shared = SharedMemory(self.ledger)
        self.rules = NewsInsightGovernanceRules(
            evidence=self.evidence,
            allowed_source_classes={"regulator", "company_filing", "reputable_newswire"},
            min_evidence_count=2,
        )
        self.gate = GovernanceGate(
            permissions=ProposePermissions({AGENT: frozenset({("insights", "claim")})}),
            evidence=self.evidence,
            schema_check=lambda resource, field, value: None,
            shared=self.shared,
            ledger=self.ledger,
            rules=(self.rules,),
        )

    def proposal(
        self,
        proposal_id: str,
        claim: str,
        direction: Direction = Direction.BULLISH,
        *,
        refs=None,
        valid_until=LATER,
        subject="ABC",
        confidence=0.7,
        finding_confidence=None,
    ) -> PromotionProposal:
        evidence_refs = refs or (
            self.evidence.add(f"{proposal_id}-1"),
            self.evidence.add(f"{proposal_id}-2", "company_filing"),
        )
        finding = Finding(
            agent="news_analyst",
            subject=subject,
            claim=claim,
            direction=direction,
            confidence=confidence if finding_confidence is None else finding_confidence,
            horizon="7 days",
            evidence_refs=tuple(ref.value for ref in evidence_refs),
        )
        return PromotionProposal(
            id=ProposalId(proposal_id),
            agent_id=AGENT,
            target_resource="insights",
            target_field="claim",
            entity_id=ENTITY,
            proposed_value=InsightRevision(
                insight_id=InsightId(proposal_id), value=finding, valid_until=valid_until
            ),
            evidence_refs=evidence_refs,
            confidence=confidence,
            reasoning_summary="Deterministic news governance test",
            created_at=CreatedAt(NOW.value),
        )

    def test_quality_news_insight_reaches_shared_memory(self) -> None:
        decision = self.gate.evaluate(self.proposal("p1", "Demand increased"), simulation_time=NOW)
        self.assertEqual(decision.outcome, GovernanceOutcome.APPROVED)
        self.assertEqual(len(self.shared.insight_history(InsightId("p1"))), 1)

    def test_insufficient_evidence_is_rejected_without_shared_write(self) -> None:
        ref = self.evidence.add("only-one")
        proposal = self.proposal("p2", "Demand increased", refs=(ref,))
        decision = self.gate.evaluate(proposal, simulation_time=NOW)
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("insufficient_news_evidence", decision.reasons)
        self.assertEqual(self.shared.insight_history(InsightId("p2")), ())

    def test_unapproved_or_missing_source_class_is_rejected(self) -> None:
        unsupported = self.evidence.add("blog", "anonymous_blog")
        unclassified = self.evidence.add("unknown", None)
        decision = self.gate.evaluate(
            self.proposal("p3", "Demand increased", refs=(unsupported, unclassified)),
            simulation_time=NOW,
        )
        self.assertIn("news_source_class_not_allowed", decision.reasons)
        self.assertIn("news_source_class_missing", decision.reasons)
        self.assertEqual(self.shared.insight_history(InsightId("p3")), ())

    def test_expiry_is_required(self) -> None:
        decision = self.gate.evaluate(
            self.proposal("p4", "Demand increased", valid_until=None), simulation_time=NOW
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("news_insight_expiry_required", decision.reasons)

    def test_semantic_duplicate_is_rejected(self) -> None:
        first = self.gate.evaluate(self.proposal("p5", "Demand increased!"), simulation_time=NOW)
        second = self.gate.evaluate(self.proposal("p6", " demand--increased "), simulation_time=NOW)
        self.assertEqual(first.outcome, GovernanceOutcome.APPROVED)
        self.assertEqual(second.outcome, GovernanceOutcome.REJECTED)
        self.assertIn("duplicate_news_insight", second.reasons)

    def test_opposing_direction_is_approved_and_tagged(self) -> None:
        self.gate.evaluate(
            self.proposal("p7", "Demand may strengthen", Direction.BULLISH), simulation_time=NOW
        )
        decision = self.gate.evaluate(
            self.proposal("p8", "Margins may weaken", Direction.BEARISH), simulation_time=NOW
        )
        self.assertEqual(decision.outcome, GovernanceOutcome.APPROVED)
        self.assertIn("conflicting_news_direction", decision.tags)

    def test_mismatched_entity_confidence_or_evidence_is_rejected(self) -> None:
        refs = (self.evidence.add("m1"), self.evidence.add("m2"))
        proposal = self.proposal(
            "p9", "Demand increased", refs=refs, subject="XYZ", finding_confidence=0.6
        )
        decision = self.gate.evaluate(proposal, simulation_time=NOW)
        self.assertIn("news_insight_entity_mismatch", decision.reasons)
        self.assertIn("news_confidence_mismatch", decision.reasons)


if __name__ == "__main__":
    unittest.main()
