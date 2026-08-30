from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRECTORY = PROJECT_ROOT / ".qfirm-cache" / "runs"
NUMBER = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"

_WALK = re.compile(r"^walk (\S+) tick=(\d+)/(\d+)$")
_ACCOUNT = re.compile(
    rf"^(?:day|book) (\S+) equity=({NUMBER}) pnl=({NUMBER}) ret=({NUMBER})% "
    rf"cash=({NUMBER}) realized=({NUMBER}) unrealized=({NUMBER}) "
    rf"fees=({NUMBER}) slip=({NUMBER})(?: (.*))?$",
    re.IGNORECASE,
)
_FILL = re.compile(
    rf"^fill (\S+) (\S+) (buy|sell) qty=({NUMBER}) px=({NUMBER}) "
    rf"fee=({NUMBER}) slip=({NUMBER})$",
    re.IGNORECASE,
)
_INSIGHT = re.compile(
    rf"^insight (\S+) (APPROVED|REJECTED) (bullish|bearish|neutral) ({NUMBER})$"
)
_COMPANY = re.compile(rf"^company (\S+) conf=({NUMBER}) (.+)$")
_THESIS_RISK = re.compile(
    rf"^thesis-risk (\S+) score=({NUMBER}) success=({NUMBER})% fail=({NUMBER})%$"
)
_REVIEW = re.compile(
    rf"^(portfolio|risk) (\S+) (approve|reject|resize|defer|allow|reduce)"
    rf"(?: size=({NUMBER}))?\s*(.*)$"
)
_HOLD = re.compile(rf"^hold (\S+) (\S+) working pnl=({NUMBER})%$")
_LOADED = re.compile(r"^loaded (\d+) Yahoo prints for (.+)$")
_SUMMARY = re.compile(r"^([a-z][a-z ]+):\s*(.+)$", re.IGNORECASE)


def latest_run_log() -> Path | None:
    configured = os.getenv("QFIRM_RUN_LOG")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_file() else None
    if not DEFAULT_RUN_DIRECTORY.is_dir():
        return None
    logs = tuple(DEFAULT_RUN_DIRECTORY.glob("*.log"))
    return max(logs, key=lambda item: item.stat().st_mtime_ns) if logs else None


def load_latest_run() -> dict[str, object] | None:
    path = latest_run_log()
    if path is None:
        return None
    stat = path.stat()
    return _parse_cached(str(path), stat.st_size, stat.st_mtime_ns)


def load_featured_trajectory() -> dict[str, object] | None:
    configured = os.getenv("QFIRM_FEATURED_RUN_LOG")
    if configured:
        candidates = [Path(configured).expanduser().resolve()]
    elif DEFAULT_RUN_DIRECTORY.is_dir():
        candidates = sorted(
            (
                path for path in DEFAULT_RUN_DIRECTORY.glob("*.log")
                if "new-rules" not in path.name
            ),
            key=lambda path: ("old-rules" in path.name, len(path.name)),
        )
    else:
        candidates = []
    for path in candidates:
        if not path.is_file():
            continue
        stat = path.stat()
        run = _parse_cached(str(path), stat.st_size, stat.st_mtime_ns)
        curve = run.get("equity_curve", [])
        if not curve:
            continue
        year = str(curve[0]["date"])[:4]
        start, end = f"{year}-01-01", f"{year}-02-28"
        points = [point for point in curve if start <= str(point["date"]) <= end]
        if not points or str(points[-1]["date"]) < f"{year}-02-27":
            continue
        january = [point for point in points if str(point["date"]) <= f"{year}-01-31"]
        peak = max(points, key=lambda point: float(point.get("return", 0)))
        milestones = [
            {**points[0], "label": "Jan 1"},
            {**january[-1], "label": "Jan close"},
            {**peak, "label": "Peak"},
            {**points[-1], "label": "Feb close"},
        ]
        return {
            "source": path.name,
            "start": start,
            "end": end,
            "points": points,
            "milestones": milestones,
        }
    return None


@lru_cache(maxsize=8)
def _parse_cached(path_value: str, size: int, modified_ns: int) -> dict[str, object]:
    del size
    path = Path(path_value)
    content = path.read_bytes()
    encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    lines = content.decode(encoding, errors="replace").splitlines()
    modified = datetime.fromtimestamp(modified_ns / 1_000_000_000, tz=timezone.utc)
    return parse_run_lines(lines, source=path.name, modified=modified)


def parse_run_lines(
    lines: list[str], *, source: str, modified: datetime
) -> dict[str, object]:
    equity_by_day: dict[str, dict[str, object]] = {}
    latest_account: dict[str, object] = {}
    positions: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    insights: list[dict[str, object]] = []
    company_notes: list[dict[str, object]] = []
    risk_notes: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    progress: dict[str, object] = {"current": 0, "total": 0, "percent": 0.0}
    summary: dict[str, object] = {}
    universe: list[str] = []
    market_prints = 0
    models: dict[str, str] = {}
    current_event_time: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("models: "):
            for key, value in re.findall(r"(news|decision)=([^\s]+)", line):
                models[f"{key}_model"] = value
            continue
        loaded = _LOADED.match(line)
        if loaded:
            market_prints = int(loaded.group(1))
            universe = [item.strip() for item in loaded.group(2).split(",") if item.strip()]
            continue
        walk = _WALK.match(line)
        if walk:
            current, total = int(walk.group(2)), int(walk.group(3))
            current_event_time = walk.group(1)
            progress = {
                "current": current,
                "total": total,
                "percent": round((current / total) * 100, 1) if total else 0.0,
                "simulation_date": walk.group(1),
            }
            continue
        account = _ACCOUNT.match(line)
        if account:
            timestamp = account.group(1)
            current_event_time = timestamp
            latest_account = {
                "timestamp": timestamp,
                "value": float(account.group(2)),
                "pnl": float(account.group(3)),
                "return": float(account.group(4)) / 100,
                "cash": float(account.group(5)),
                "realized_pnl": float(account.group(6)),
                "unrealized_pnl": float(account.group(7)),
                "fees": float(account.group(8)),
                "slippage": float(account.group(9)),
            }
            day = timestamp[:10]
            equity_by_day[day] = {
                "date": day,
                "value": latest_account["value"],
                "return": latest_account["return"],
            }
            positions = _parse_positions(account.group(10) or "")
            continue
        fill = _FILL.match(line)
        if fill:
            item = {
                "timestamp": fill.group(1),
                "symbol": fill.group(2),
                "side": fill.group(3).lower(),
                "quantity": float(fill.group(4)),
                "price": float(fill.group(5)),
                "fee": float(fill.group(6)),
                "slippage": float(fill.group(7)),
            }
            fills.append(item)
            audit.append(_audit_item("fill", item["timestamp"], item["symbol"], item["side"]))
            current_event_time = item["timestamp"]
            continue
        insight = _INSIGHT.match(line)
        if insight:
            item = {
                "symbol": insight.group(1),
                "outcome": insight.group(2).lower(),
                "direction": insight.group(3),
                "confidence": float(insight.group(4)),
                "timestamp": current_event_time,
            }
            insights.append(item)
            if item["outcome"] == "approved":
                audit.append(_audit_item("insight", current_event_time, item["symbol"], "approved"))
            continue
        company = _COMPANY.match(line)
        if company:
            company_notes.append({
                "symbol": company.group(1),
                "confidence": float(company.group(2)),
                "summary": company.group(3),
                "timestamp": current_event_time,
            })
            continue
        risk = _THESIS_RISK.match(line)
        if risk:
            item = {
                "symbol": risk.group(1),
                "score": float(risk.group(2)),
                "success_probability": float(risk.group(3)) / 100,
                "failure_probability": float(risk.group(4)) / 100,
                "timestamp": current_event_time,
            }
            risk_notes.append(item)
            continue
        review = _REVIEW.match(line)
        if review:
            item = {
                "type": review.group(1),
                "symbol": review.group(2),
                "action": review.group(3),
                "size": float(review.group(4)) if review.group(4) else None,
                "rationale": review.group(5).strip(),
                "timestamp": current_event_time,
            }
            reviews.append(item)
            audit.append(_audit_item(item["type"], current_event_time, item["symbol"], item["action"]))
            continue
        hold = _HOLD.match(line)
        if hold:
            item = {
                "type": "position",
                "symbol": hold.group(1),
                "action": hold.group(2),
                "pnl": float(hold.group(3)) / 100,
                "timestamp": current_event_time,
                "rationale": "Existing paper position reviewed against current evidence.",
            }
            reviews.append(item)
            audit.append(_audit_item("position", current_event_time, item["symbol"], item["action"]))
            continue
        summary_match = _SUMMARY.match(line)
        if summary_match:
            _capture_summary(summary, summary_match.group(1).lower(), summary_match.group(2))

    curve = list(equity_by_day.values())
    curve = _downsample(curve, 180)
    drawdown = _max_drawdown([float(point["value"]) for point in curve])
    if "max_drawdown" in summary:
        drawdown = float(summary["max_drawdown"])

    age_seconds = max(0.0, (datetime.now(timezone.utc) - modified).total_seconds())
    completed = "end_equity" in summary
    status = "completed" if completed else ("running" if age_seconds < 900 else "paused")
    if completed:
        progress["percent"] = 100.0

    return {
        "id": source.removesuffix(".log"),
        "source": source,
        "status": status,
        "updated_at": modified.isoformat(),
        "progress": progress,
        "window": _window_from_name(source),
        "market": {
            "universe": universe,
            "prints": market_prints,
            "interval": _interval_from_name(source),
        },
        "portfolio": {**latest_account, "positions": positions},
        "metrics": {**summary, "max_drawdown": drawdown},
        "equity_curve": curve,
        "fills": fills[-40:][::-1],
        "research": {
            "insights": insights[-60:][::-1],
            "approved": sum(item["outcome"] == "approved" for item in insights),
            "rejected": sum(item["outcome"] == "rejected" for item in insights),
            "companies": company_notes[-20:][::-1],
            "risks": risk_notes[-20:][::-1],
        },
        "reviews": reviews[-40:][::-1],
        "audit": audit[-80:][::-1],
        "models": models,
    }


def _parse_positions(value: str) -> list[dict[str, object]]:
    positions = []
    for item in value.split(","):
        match = re.fullmatch(r"([A-Z.\-]+)([-+]" + NUMBER.lstrip("[-+]?") + r")", item.strip(), re.IGNORECASE)
        if match:
            positions.append({"symbol": match.group(1).upper(), "quantity": float(match.group(2))})
    return positions


def _capture_summary(summary: dict[str, object], label: str, value: str) -> None:
    names = {
        "end equity": ("end_equity", float),
        "return": ("return", lambda item: float(item.rstrip("%")) / 100),
        "pnl": ("pnl", float),
        "max drawdown": ("max_drawdown", lambda item: float(item.rstrip("%")) / 100),
        "average gross exposure": ("gross_exposure", lambda item: float(item.rstrip("x"))),
        "turnover": ("turnover", lambda item: float(item.rstrip("x"))),
        "transaction cost": ("transaction_cost", float),
        "sharpe": ("sharpe", float),
        "hit rate": ("hit_rate", lambda item: float(item.rstrip("%")) / 100),
        "profit factor": ("profit_factor", float),
    }
    if label not in names or value == "undefined":
        return
    key, parser = names[label]
    try:
        summary[key] = parser(value)
    except ValueError:
        return


def _audit_item(kind: str, timestamp: object, symbol: object, state: object) -> dict[str, object]:
    return {"type": kind, "timestamp": timestamp, "symbol": symbol, "state": state}


def _max_drawdown(values: list[float]) -> float:
    peak = 0.0
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            maximum = max(maximum, (peak - value) / peak)
    return maximum


def _downsample(points: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(index * step)] for index in range(limit)]


def _window_from_name(source: str) -> dict[str, str] | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", source)
    return {"start": match.group(1), "end": match.group(2)} if match else None


def _interval_from_name(source: str) -> str | None:
    match = re.search(r"_((?:1|5|15)m|1h|1d)(?:\.|$)", source)
    return match.group(1) if match else None


def write_snapshot(payload: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(destination)
