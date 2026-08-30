from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agents.model import resolve_decision_model, resolve_news_model
from agents.registry import REGISTRY

WEB_ROOT = Path(__file__).resolve().parent


def dashboard_payload() -> dict[str, object]:
    agents = []
    for key in REGISTRY.keys():
        spec = REGISTRY.get(key)
        agents.append({
            "name": spec.name,
            "version": spec.version,
            "output": str(spec.config.get("output") or spec.config.get("role") or spec.config.get("source") or "deterministic"),
        })
    return {
        "system": {"status": "Operational", "mode": "Research and paper trading"},
        "agents": agents,
        "run": None,
        "settings": {
            "news_model": resolve_news_model(),
            "decision_model": resolve_decision_model(),
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/api/dashboard":
            body = json.dumps(dashboard_payload(), separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
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
