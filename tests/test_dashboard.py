import unittest

from web.server import dashboard_payload


class DashboardPayloadTests(unittest.TestCase):
    def test_exposes_registry_without_secret(self):
        payload = dashboard_payload()
        self.assertGreaterEqual(len(payload["agents"]), 1)
        self.assertIsNone(payload["run"])
        self.assertNotIn("OPENAI_API_KEY", str(payload))


if __name__ == "__main__":
    unittest.main()
