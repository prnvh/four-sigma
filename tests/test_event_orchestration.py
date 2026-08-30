import unittest
from datetime import datetime, timedelta, timezone

from agents import AgentTrigger, EventDrivenOrchestrator, TriggerType


NOW = datetime(2026, 8, 1, tzinfo=timezone.utc)


def trigger(kind, entity="ABC", minute=0, ref=None):
    return AgentTrigger(kind, entity, NOW + timedelta(minutes=minute), ref)


class EventDrivenOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = EventDrivenOrchestrator({
            "research": {
                TriggerType.RELEVANT_NEWS,
                TriggerType.NEW_FILING,
                TriggerType.LARGE_PRICE_MOVE,
                TriggerType.SCHEDULED_REVIEW,
            },
            "portfolio": {
                TriggerType.PORTFOLIO_CHANGE,
                TriggerType.INSIGHT_EXPIRY,
            },
        })

    def test_no_event_means_no_agent_run(self):
        self.assertEqual(
            self.orchestrator.plan((), simulation_time=NOW), ()
        )

    def test_coalesces_many_events_into_one_run_per_agent_and_entity(self):
        plans = self.orchestrator.plan(
            (
                trigger(TriggerType.RELEVANT_NEWS, ref="news:1"),
                trigger(TriggerType.RELEVANT_NEWS, ref="news:2"),
                trigger(TriggerType.NEW_FILING, ref="filing:1"),
            ),
            simulation_time=NOW,
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].agent_key, "research")
        self.assertEqual(
            plans[0].triggers,
            (TriggerType.NEW_FILING, TriggerType.RELEVANT_NEWS),
        )
        self.assertEqual(plans[0].source_refs, ("filing:1", "news:1", "news:2"))

    def test_routes_portfolio_and_research_triggers_to_different_work(self):
        plans = self.orchestrator.plan(
            (
                trigger(TriggerType.LARGE_PRICE_MOVE),
                trigger(TriggerType.PORTFOLIO_CHANGE, ref="fill:1"),
                trigger(TriggerType.INSIGHT_EXPIRY, ref="insight:1"),
            ),
            simulation_time=NOW,
        )
        self.assertEqual([item.agent_key for item in plans], ["portfolio", "research"])
        self.assertEqual(
            plans[0].triggers,
            (TriggerType.INSIGHT_EXPIRY, TriggerType.PORTFOLIO_CHANGE),
        )

    def test_future_trigger_cannot_schedule_work_early(self):
        plans = self.orchestrator.plan(
            (trigger(TriggerType.RELEVANT_NEWS, minute=1),),
            simulation_time=NOW,
        )
        self.assertEqual(plans, ())

    def test_entities_are_normalized_and_plans_are_deterministic(self):
        events = (
            trigger(TriggerType.SCHEDULED_REVIEW, " xyz "),
            trigger(TriggerType.SCHEDULED_REVIEW, "abc"),
        )
        first = self.orchestrator.plan(events, simulation_time=NOW)
        second = self.orchestrator.plan(tuple(reversed(events)), simulation_time=NOW)
        self.assertEqual(first, second)
        self.assertEqual([item.entity_id for item in first], ["ABC", "XYZ"])

    def test_all_six_roadmap_triggers_are_supported(self):
        self.assertEqual(
            set(TriggerType),
            {
                TriggerType.RELEVANT_NEWS,
                TriggerType.NEW_FILING,
                TriggerType.LARGE_PRICE_MOVE,
                TriggerType.PORTFOLIO_CHANGE,
                TriggerType.INSIGHT_EXPIRY,
                TriggerType.SCHEDULED_REVIEW,
            },
        )

    def test_invalid_orchestration_inputs_fail_safely(self):
        with self.assertRaises(ValueError):
            EventDrivenOrchestrator({})
        with self.assertRaises(ValueError):
            AgentTrigger(TriggerType.NEW_FILING, "", NOW)
        with self.assertRaises(ValueError):
            AgentTrigger(TriggerType.NEW_FILING, "ABC", datetime(2026, 1, 1))
        with self.assertRaises(TypeError):
            self.orchestrator.plan(("not-an-event",), simulation_time=NOW)


if __name__ == "__main__":
    unittest.main()
