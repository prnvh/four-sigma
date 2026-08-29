import unittest

from memory import AgentId, CanonicalId, EventId, RunId


class IdTests(unittest.TestCase):
    def test_distinct_kinds_are_not_interchangeable(self) -> None:
        agent = AgentId("news")
        event = EventId("news")
        self.assertNotEqual(agent, event)
        self.assertIsInstance(agent, AgentId)
        self.assertNotIsInstance(event, AgentId)

    def test_same_kind_compares_by_value(self) -> None:
        self.assertEqual(RunId("r1"), RunId("r1"))
        self.assertNotEqual(RunId("r1"), RunId("r2"))

    def test_empty_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgentId("")
        with self.assertRaises(ValueError):
            AgentId("   ")

    def test_base_type_cannot_be_constructed(self) -> None:
        with self.assertRaises(TypeError):
            CanonicalId("x")


if __name__ == "__main__":
    unittest.main()
