import threading
import unittest
from datetime import datetime, timezone

from agents import (
    ConcurrentAgentExecutor,
    PlannedAgentRun,
    TriggerType,
)


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def plan(agent, entity):
    return PlannedAgentRun(
        agent_key=agent,
        entity_id=entity,
        simulation_time=NOW,
        triggers=(TriggerType.RELEVANT_NEWS,),
        source_refs=(f"news:{entity}",),
    )


class ConcurrentAgentExecutorTests(unittest.TestCase):
    def test_safe_compute_runs_in_parallel(self):
        barrier = threading.Barrier(2)
        worker_threads = set()
        lock = threading.Lock()

        def compute(item):
            with lock:
                worker_threads.add(threading.get_ident())
            barrier.wait(timeout=2)
            return item.entity_id.lower()

        result = ConcurrentAgentExecutor(max_workers=2).execute(
            (plan("news", "ABC"), plan("company", "XYZ")),
            compute=compute,
        )
        self.assertEqual(len(result.succeeded), 2)
        self.assertEqual(len(worker_threads), 2)

    def test_commits_are_serial_deterministic_and_after_all_compute(self):
        main_thread = threading.get_ident()
        computed = []
        committed = []
        lock = threading.Lock()

        def compute(item):
            with lock:
                computed.append(item.entity_id)
            return item.entity_id

        def commit(result):
            self.assertEqual(threading.get_ident(), main_thread)
            self.assertEqual(len(computed), 3)
            committed.append(result.plan.entity_id)

        ConcurrentAgentExecutor(max_workers=3).execute(
            (
                plan("research", "XYZ"),
                plan("research", "ABC"),
                plan("research", "MNO"),
            ),
            compute=compute,
            commit=commit,
        )
        self.assertEqual(committed, ["ABC", "MNO", "XYZ"])

    def test_failed_compute_is_isolated_and_never_committed(self):
        committed = []

        def compute(item):
            if item.entity_id == "BAD":
                raise RuntimeError("synthetic failure")
            return "validated"

        result = ConcurrentAgentExecutor().execute(
            (plan("research", "GOOD"), plan("research", "BAD")),
            compute=compute,
            commit=lambda item: committed.append(item.plan.entity_id),
        )
        self.assertEqual(committed, ["GOOD"])
        self.assertEqual([item.plan.entity_id for item in result.failed], ["BAD"])
        self.assertEqual(result.failed[0].error_type, "RuntimeError")

    def test_input_order_does_not_change_result_order(self):
        plans = (plan("z-agent", "XYZ"), plan("a-agent", "ABC"))
        first = ConcurrentAgentExecutor().execute(plans, compute=lambda item: item.entity_id)
        second = ConcurrentAgentExecutor().execute(
            tuple(reversed(plans)), compute=lambda item: item.entity_id
        )
        self.assertEqual(first, second)

    def test_empty_batch_does_no_work(self):
        called = False

        def compute(item):
            nonlocal called
            called = True

        self.assertEqual(
            ConcurrentAgentExecutor().execute((), compute=compute).succeeded, ()
        )
        self.assertFalse(called)

    def test_duplicate_plan_is_rejected_before_compute(self):
        item = plan("research", "ABC")
        with self.assertRaisesRegex(ValueError, "duplicate planned"):
            ConcurrentAgentExecutor().execute((item, item), compute=lambda value: value)

    def test_worker_limit_and_input_types_are_validated(self):
        with self.assertRaises(ValueError):
            ConcurrentAgentExecutor(0)
        with self.assertRaises(TypeError):
            ConcurrentAgentExecutor(True)
        with self.assertRaises(TypeError):
            ConcurrentAgentExecutor().execute(("bad-plan",), compute=lambda value: value)


if __name__ == "__main__":
    unittest.main()
