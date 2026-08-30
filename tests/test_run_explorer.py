import unittest
from datetime import timedelta

from agents import RunExplorer, RunExplorerError, run_backtest
from memory import (
    AgentId,
    AuditLedger,
    ContextGateway,
    ContextSnapshotStore,
    CreatedAt,
    EntityId,
    EvidenceId,
    InsightId,
    InsightRevision,
    PromotionProposal,
    ProposalId,
    SharedMemory,
    SimulationTime,
)
from tests.test_backtest import DAY1, DAY3, insight, store, tape


class RunExplorerTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = ContextSnapshotStore()
        gateway = ContextGateway(store(insight()), snapshots=self.snapshots)
        self.result = run_backtest(
            start=DAY1,
            end=DAY3,
            universe=("ABC",),
            agent_versions=("trade_constructor:v1",),
            strategy_config={
                "starting_cash": 1000,
                "step": timedelta(days=1),
                "slippage_bps": 0,
                "fee_bps": 0,
                "max_position_pct": 0.1,
            },
            store=gateway,
            tape=tape(),
        )
        self.explorer = RunExplorer(self.result, context_store=self.snapshots)

    def test_explains_trade_from_candidate_audit_and_fill_records(self):
        explanation = self.explorer.why_trade("abc")
        self.assertEqual(explanation.instrument, "ABC")
        self.assertEqual(explanation.direction, "long")
        self.assertEqual(explanation.thesis_refs, ("insight:1",))
        self.assertEqual(explanation.fills[0].instrument, "ABC")
        self.assertEqual(
            [item.to_status for item in explanation.lifecycle],
            ["proposed", "risk_reviewed", "approved", "submitted", "filled", "closed"],
        )
        self.assertTrue(explanation.context_snapshot_ids)

    def test_returns_exact_stored_context_view(self):
        snapshot_id = self.result.context_snapshots[0].id.value
        explanation = self.explorer.context(snapshot_id)
        self.assertEqual(explanation.snapshot.id.value, snapshot_id)
        self.assertEqual(explanation.view["symbol"], "ABC")
        self.assertEqual(explanation.view["promoted_insights"][0]["ref"], "insight:1")

    def test_explains_audited_size_review(self):
        candidate_id = self.result.candidates[0].id.value
        explanation = self.explorer.why_resized(candidate_id)
        self.assertEqual(explanation.original_size, 1.0)
        self.assertEqual(explanation.reviewed_size, 0.1)
        self.assertTrue(explanation.resized)
        self.assertTrue(explanation.reason)

    def test_resolves_insight_version_to_creating_proposal(self):
        shared = SharedMemory(AuditLedger())
        shared.apply_approved(
            PromotionProposal(
                id=ProposalId("proposal:1"),
                agent_id=AgentId("news_analyst"),
                target_resource="insights",
                target_field="claim",
                entity_id=EntityId("ABC"),
                proposed_value=InsightRevision(
                    insight_id=InsightId("insight:x"), value="Synthetic insight"
                ),
                evidence_refs=(EvidenceId("news:1"),),
                confidence=0.8,
                reasoning_summary="Synthetic proposal",
                created_at=CreatedAt(DAY1),
            ),
            decided_at=SimulationTime(DAY1),
        )
        origin = RunExplorer(self.result, shared_memory=shared).insight_origin("insight:x")
        self.assertEqual(origin.proposal_id, "proposal:1")
        self.assertEqual(origin.version, 1)

    def test_unknown_records_and_missing_sources_fail_explicitly(self):
        with self.assertRaises(RunExplorerError):
            self.explorer.why_trade("MISSING")
        with self.assertRaisesRegex(RunExplorerError, "view store"):
            RunExplorer(self.result).context(self.result.context_snapshots[0].id.value)
        with self.assertRaisesRegex(RunExplorerError, "shared memory"):
            self.explorer.insight_origin("insight:1")


if __name__ == "__main__":
    unittest.main()
