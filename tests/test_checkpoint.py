import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from memory import CheckpointError, CheckpointStore, SimulationCheckpoint


NOW = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)


def checkpoint():
    return SimulationCheckpoint(
        simulation_time=NOW,
        portfolio_state={"cash": 900.0, "positions": [{"symbol": "ABC", "qty": 1}]},
        shared_memory={"risk": {"ABC": {"score": 40}}},
        working_memory=[{"agent": "risk_llm", "entity": "ABC", "value": "review"}],
        audit_offset=42,
        pending_events=[{"type": "fill", "symbol": "ABC", "at": NOW}],
    )


class CheckpointTests(unittest.TestCase):
    def test_persists_and_resumes_all_required_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            path = store.save("month-01", checkpoint())
            self.assertTrue(path.exists())
            restored = store.load("month-01")
            resumed = restored.resume()
            self.assertEqual(resumed.clock.now().value, NOW)
            self.assertEqual(resumed.portfolio_state["cash"], 900.0)
            self.assertEqual(resumed.shared_memory["risk"]["ABC"]["score"], 40)
            self.assertEqual(resumed.working_memory[0]["agent"], "risk_llm")
            self.assertEqual(resumed.audit_offset, 42)
            self.assertEqual(resumed.pending_events[0]["type"], "fill")
            self.assertEqual(
                resumed.pending_events[0]["at"], "2026-06-01T12:00:00+00:00"
            )

    def test_checksum_rejects_tampered_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            path = store.save("safe", checkpoint())
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["payload"]["portfolio_state"]["cash"] = 999999
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(CheckpointError, "checksum mismatch"):
                store.load("safe")

    def test_malformed_or_partial_checkpoint_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.json"
            path.write_text('{"payload":{}}', encoding="utf-8")
            with self.assertRaises(CheckpointError):
                CheckpointStore(directory).load("partial")

    def test_name_cannot_escape_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(directory)
            for name in ("../escape", "nested/file", ""):
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        store.save(name, checkpoint())

    def test_checkpoint_rejects_naive_time_and_negative_audit_offset(self):
        state = checkpoint()
        with self.assertRaises(ValueError):
            SimulationCheckpoint(
                simulation_time=NOW.replace(tzinfo=None),
                portfolio_state=state.portfolio_state,
                shared_memory=state.shared_memory,
                working_memory=state.working_memory,
                audit_offset=0,
                pending_events=state.pending_events,
            )
        with self.assertRaises(ValueError):
            SimulationCheckpoint(
                simulation_time=NOW,
                portfolio_state={}, shared_memory={}, working_memory=[],
                audit_offset=-1, pending_events=[],
            )


if __name__ == "__main__":
    unittest.main()
