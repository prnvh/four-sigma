from __future__ import annotations

import argparse
import hashlib
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.model import resolve_decision_model, resolve_news_model
from agents.registry import REGISTRY
from memory.loss_guard import LossGuardLimits
from memory.position_risk import PositionRiskLimits

from .run_snapshot import load_featured_trajectory, load_latest_run

WEB_ROOT = Path(__file__).resolve().parent


def dashboard_payload() -> dict[str, object]:
    try:
        run = load_latest_run()
    except (OSError, ValueError):
        run = None
    try:
        featured_trajectory = load_featured_trajectory()
    except (OSError, ValueError):
        featured_trajectory = None
    agents = []
    for key in REGISTRY.keys():
        spec = REGISTRY.get(key)
        agents.append({
            "name": spec.name,
            "version": spec.version,
            "output": str(spec.config.get("output") or spec.config.get("role") or spec.config.get("source") or "deterministic"),
        })
    position_limits = PositionRiskLimits()
    loss_limits = LossGuardLimits()
    activity = [
        {"icon": "R", "title": "Agent registry synchronized", "detail": f"{len(agents)} versioned specifications loaded", "state": "Verified"},
        {"icon": "G", "title": "Deterministic risk gate active", "detail": "Model outputs cannot override vetoes", "state": "Enforced"},
        {"icon": "P", "title": "Execution authority restricted", "detail": "Research and paper trading only", "state": "Safe"},
    ]
    if run:
        progress = run.get("progress", {})
        activity.insert(0, {
            "icon": "L" if run.get("status") == "running" else "✓",
            "title": f"Paper run {run.get('status', 'connected')}",
            "detail": f"{progress.get('percent', 0):.1f}% · {run.get('source')}",
            "state": "Live" if run.get("status") == "running" else "Loaded",
        })
    return {
        "system": {
            "status": "Operational",
            "mode": "Live local paper run" if run and run.get("status") == "running" else "Research and paper trading",
            "connected": bool(run),
        },
        "agents": agents,
        "run": run,
        "featured_trajectory": featured_trajectory,
        "risk_limits": [
            {
                "label": "Portfolio drawdown",
                "current": _percent(run, "max_drawdown"),
                "limit": f"{loss_limits.max_portfolio_drawdown:.0%}",
                "utilization": _utilization(run, "max_drawdown", loss_limits.max_portfolio_drawdown),
            },
            {"label": "Single position", "current": "Live check", "limit": f"{position_limits.max_position_pct:.0%}", "utilization": None},
            {"label": "Annualized volatility", "current": "Live check", "limit": f"{position_limits.max_annualized_volatility:.0%}", "utilization": None},
            {
                "label": "Gross exposure",
                "current": _multiple(run, "gross_exposure"),
                "limit": f"{position_limits.max_gross_exposure:.2f}x",
                "utilization": _utilization(run, "gross_exposure", position_limits.max_gross_exposure),
            },
        ],
        "activity": activity,
        "pipeline": [
            {"step": "01", "name": "Market + news intake", "detail": "Point-in-time Yahoo bars and sourced GDELT events"},
            {"step": "02", "name": "Evidence analysis", "detail": "News, company and scenario-risk agents"},
            {"step": "03", "name": "Governance", "detail": "Schema, permission and evidence promotion gates"},
            {"step": "04", "name": "Deterministic risk", "detail": "Concentration, liquidity, volatility and drawdown vetoes"},
            {"step": "05", "name": "Paper execution", "detail": "Simulated fills with fees, slippage and complete audit lineage"},
        ],
        "controls": [
            {"name": "Position cap", "value": f"{position_limits.max_position_pct:.0%}", "effect": "Resize or reject"},
            {"name": "Gross exposure", "value": f"{position_limits.max_gross_exposure:.2f}x", "effect": "Hard ceiling"},
            {"name": "Portfolio loss guard", "value": f"{loss_limits.max_portfolio_drawdown:.0%}", "effect": "Required exit"},
            {"name": "Position stop", "value": f"{loss_limits.max_position_loss:.0%}", "effect": "Required exit"},
            {"name": "Loss cooldown", "value": f"{loss_limits.loss_cooldown.days} days", "effect": "Blocks re-entry"},
            {"name": "Averaging down", "value": "Disabled", "effect": "Fail closed"},
        ],
        "settings": {
            "news_model": resolve_news_model(),
            "decision_model": resolve_decision_model(),
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


def _metric(run: dict[str, object] | None, name: str) -> float | None:
    if not run:
        return None
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _percent(run: dict[str, object] | None, name: str) -> str:
    value = _metric(run, name)
    return "Awaiting run" if value is None else f"{value:.2%}"


def _multiple(run: dict[str, object] | None, name: str) -> str:
    value = _metric(run, name)
    return "Awaiting completion" if value is None else f"{value:.2f}x"


def _utilization(run: dict[str, object] | None, name: str, limit: float) -> float | None:
    value = _metric(run, name)
    return None if value is None or not limit else min(1.0, max(0.0, value / limit))


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/api/dashboard":
            body = json.dumps(dashboard_payload(), separators=(",", ":")).encode()
            etag = f'"{hashlib.sha256(body).hexdigest()[:20]}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the 4 Sigma dashboard and read-only API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"4 Sigma dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
