import tempfile
import unittest
from pathlib import Path

from agents.model import CachedModelClient


class CountingModel:
    model_name = "synthetic-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, *, instructions, input_data, schema):
        self.calls += 1
        return {"call": self.calls, "value": input_data["value"]}


class CachedModelClientTests(unittest.TestCase):
    def test_identical_request_is_replayed_without_another_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = CountingModel()
            cached = CachedModelClient(model, Path(directory) / "responses.sqlite3")
            request = {
                "instructions": "Return structured data.",
                "input_data": {"value": 7},
                "schema": {"type": "object"},
            }
            first = cached.generate_json(**request)
            first["value"] = 99
            second = cached.generate_json(**request)
            self.assertEqual(model.calls, 1)
            self.assertEqual(second, {"call": 1, "value": 7})

    def test_prompt_or_input_change_invalidates_the_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = CountingModel()
            cached = CachedModelClient(model, Path(directory) / "responses.sqlite3")
            schema = {"type": "object"}
            cached.generate_json(instructions="a", input_data={"value": 1}, schema=schema)
            cached.generate_json(instructions="b", input_data={"value": 1}, schema=schema)
            cached.generate_json(instructions="b", input_data={"value": 2}, schema=schema)
            self.assertEqual(model.calls, 3)


if __name__ == "__main__":
    unittest.main()
