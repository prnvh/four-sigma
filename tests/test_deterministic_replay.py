import unittest
from datetime import datetime, timedelta, timezone

from agents import NewsAnalyst, run_backtest
from memory import AuditEventType, Evidence, MarketTape, PricePrint, ResearchContextStore


DAY1 = datetime(2026, 4, 1, tzinfo=timezone.utc)
DAY2 = datetime(2026, 4, 2, tzinfo=timezone.utc)
DAY3 = datetime(2026, 4, 3, tzinfo=timezone.utc)


class RecordedNewsModel:
    """A replay fixture: identical context always receives the recorded response."""

    def generate_json(self, *, instructions, input_data, schema):
        refs = tuple(item["ref"] for item in input_data["articles"])
        if refs != ("news:1", "news:2"):
            raise AssertionError(f"unrecorded context: {refs}")
        return {
            "claim": "Recorded demand evidence improved",
            "direction": "bullish",
            "confidence": 0.75,
            "horizon": "three days",
            "evidence_refs": ["news:1", "news:2"],
            "risks": ["Recorded demand may reverse"],
        }


def historical_inputs():
    store = ResearchContextStore()
    for number in (1, 2):
        store.append_news(Evidence(
            ref=f"news:{number}",
            source="Recorded fixture",
            url=f"https://example.test/news/{number}",
            published_at=DAY1,
            knowledge_time=DAY1,
            title=f"Recorded headline {number}",
            summary=f"Recorded summary {number}",
            symbols=("ABC",),
        ))
    tape = MarketTape((
        PricePrint(symbol="ABC", price=10, knowledge_time=DAY1),
        PricePrint(symbol="ABC", price=12, knowledge_time=DAY2),
        PricePrint(symbol="ABC", price=13, knowledge_time=DAY3),
    ))
    return store, tape


def replay():
    store, tape = historical_inputs()
    return run_backtest(
        start=DAY1,
        end=DAY3,
        universe=("ABC",),
        agent_versions=("news_analyst:v1", "trade_constructor:v1"),
        strategy_config={
            "starting_cash": 1000,
            "step": timedelta(days=1),
            "max_position_pct": 0.1,
            "min_evidence_count": 2,
            "insight_horizon_days": 3,
        },
        store=store,
        tape=tape,
        news_analyst=NewsAnalyst(RecordedNewsModel()),
    )


class DeterministicReplayTests(unittest.TestCase):
    def test_full_historical_trace_replays_identically(self):
        first = replay()
        second = replay()

        self.assertTrue(first.context_snapshots)
        self.assertEqual(first.context_snapshots, second.context_snapshots)
        self.assertEqual(first.invocations, second.invocations)
        self.assertEqual(first.audit_events, second.audit_events)
        self.assertEqual(first.findings, second.findings)
        self.assertEqual(first.promotions, second.promotions)
        self.assertEqual(first.candidates, second.candidates)
        self.assertEqual(first.fills, second.fills)
        self.assertEqual(first.snapshots, second.snapshots)
        self.assertEqual(first.final, second.final)

        memory_types = {
            AuditEventType.WORKING_MEMORY_WRITTEN,
            AuditEventType.PROMOTION_REQUESTED,
            AuditEventType.PROMOTION_APPROVED,
            AuditEventType.SHARED_MEMORY_UPDATED,
        }
        first_memory = tuple(
            event for event in first.audit_events if event.event_type in memory_types
        )
        second_memory = tuple(
            event for event in second.audit_events if event.event_type in memory_types
        )
        self.assertTrue(first_memory)
        self.assertEqual(first_memory, second_memory)

    def test_context_hashes_are_replayable_and_point_in_time(self):
        result = replay()
        self.assertTrue(all(
            snapshot.id.value == snapshot.content_hash
            for snapshot in result.context_snapshots
        ))
        self.assertTrue(all(
            all(ref in {"news:1", "news:2"} or ref.startswith("insight:")
                for ref in snapshot.source_refs)
            for snapshot in result.context_snapshots
        ))
        self.assertTrue(all(
            snapshot.simulation_time <= DAY3 for snapshot in result.context_snapshots
        ))


if __name__ == "__main__":
    unittest.main()
