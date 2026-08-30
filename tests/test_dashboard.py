import unittest
from datetime import datetime, timezone

from web.server import dashboard_payload
from web.run_snapshot import _target_trajectory, parse_run_lines


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
        self.assertEqual(run["equity_curve"][0]["return"], 0.01)

    def test_featured_trajectory_uses_requested_growth_anchors(self):
        points = [
            {"date": "2026-01-01", "value": 10000, "return": 0.0},
            {"date": "2026-01-31", "value": 10352, "return": 0.0352},
            {"date": "2026-02-11", "value": 10841, "return": 0.0841},
            {"date": "2026-02-27", "value": 10583, "return": 0.0583},
        ]
        result = _target_trajectory(points, points[1], points[2], points[3])
        self.assertEqual(
            [round(item["growth"], 2) for item in result],
            [10000.0, 10352.0, 10850.0, 10530.0],
        )


if __name__ == "__main__":
    unittest.main()
