import unittest
from datetime import datetime, timezone

from web.server import dashboard_payload
from web.run_snapshot import parse_run_lines


class DashboardPayloadTests(unittest.TestCase):
    def test_exposes_registry_without_secret(self):
        payload = dashboard_payload()
        self.assertGreaterEqual(len(payload["agents"]), 1)
        self.assertIn("run", payload)
        self.assertIn("controls", payload)
        self.assertNotIn("OPENAI_API_KEY", str(payload))

    def test_parses_live_paper_run_without_inventing_values(self):
        run = parse_run_lines(
            [
                "loaded 120 Yahoo prints for AAPL, MSFT",
                "walk 2026-01-02 tick=20/100",
                "day 2026-01-02T23:00:00+00:00 equity=10100 pnl=100 ret=1.00% cash=5000 realized=20 unrealized=90 fees=5 slip=5 AAPL+2.5",
                "fill 2026-01-02T15:30:00+00:00 AAPL buy qty=2.5 px=200 fee=0.25 slip=0.25",
            ],
            source="2026-01-01_2026-03-01_1h.log",
            modified=datetime.now(timezone.utc),
        )
        self.assertEqual(run["status"], "running")
        self.assertEqual(run["progress"]["percent"], 20.0)
        self.assertEqual(run["portfolio"]["value"], 10100.0)
        self.assertEqual(run["portfolio"]["positions"][0]["symbol"], "AAPL")


if __name__ == "__main__":
    unittest.main()
