from __future__ import annotations

import argparse
from pathlib import Path

from .run_snapshot import PROJECT_ROOT, write_snapshot
from .server import dashboard_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the latest local paper run for a static/Vercel dashboard."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "web" / "data" / "dashboard.json",
    )
    args = parser.parse_args()
    destination = args.output.resolve()
    payload = dashboard_payload()
    payload["system"]["mode"] = "Static paper-run snapshot"
    if payload.get("run"):
        payload["run"]["status"] = "snapshot"
    write_snapshot(payload, destination)
    print(f"Dashboard snapshot written to {destination}")


if __name__ == "__main__":
    main()
