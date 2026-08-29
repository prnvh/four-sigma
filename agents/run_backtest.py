from __future__ import annotations

import argparse
from datetime import datetime, timezone

from memory.types import jsonable

from .backtest import run_backtest
from .model import OpenAIModelClient
from .news_analyst import NewsAnalyst


def _when(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end equity backtest: Yahoo bars + GDELT news + agents + paper fills."
    )
    parser.add_argument("--start", type=_when, required=True, help="UTC start, YYYY-MM-DD")
    parser.add_argument("--end", type=_when, required=True, help="UTC end, YYYY-MM-DD")
    parser.add_argument("--universe", nargs="+", required=True, help="Equity tickers, e.g. AAPL MSFT")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--interval", default="15m", choices=("1m", "5m", "15m", "1h", "1d"))
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--fee-bps", type=float, default=5)
    parser.add_argument("--max-position-pct", type=float, default=0.1)
    args = parser.parse_args()
    result = run_backtest(
        start=args.start,
        end=args.end,
        universe=args.universe,
        agent_versions=("news_analyst:v1", "trade_constructor:v1"),
        strategy_config={
            "starting_cash": args.cash,
            "interval": args.interval,
            "step": "bar",
            "slippage_bps": args.slippage_bps,
            "fee_bps": args.fee_bps,
            "max_position_pct": args.max_position_pct,
            "insight_horizon_days": 3,
            "min_evidence_count": 2,
        },
        news_analyst=NewsAnalyst(OpenAIModelClient()),
    )
    start_equity = args.cash
    print(f"window: {args.start.isoformat()} → {args.end.isoformat()}")
    print(f"interval: {args.interval}")
    print(f"marks: {len(result.snapshots)}")
    print(f"findings: {len(result.findings)}")
    print(
        "promotions approved: "
        f"{sum(1 for item in result.promotions if item.outcome.value == 'APPROVED')}"
    )
    print(f"candidates: {len(result.candidates)}")
    print(f"fills: {len(result.fills)}")
    print(f"start equity: {start_equity}")
    print(f"end equity: {result.final.equity}")
    print(f"return: {(result.final.equity / start_equity) - 1:.4%}")
    print(f"pnl: {result.final.equity - start_equity}")
    for finding in result.findings:
        print(
            f"finding {finding.subject} {finding.direction.value} "
            f"{finding.confidence:.2f} {finding.claim}"
        )
    for candidate in result.candidates:
        print(
            f"decision {candidate.knowledge_time.isoformat()} {candidate.instrument} "
            f"{candidate.direction.value} {candidate.status.value} size={candidate.proposed_size}"
        )
    for fill in result.fills:
        print(
            f"fill {fill.knowledge_time.isoformat()} {fill.instrument} {fill.side.value} "
            f"qty={fill.quantity} px={fill.price} fee={fill.fee} slip={fill.slippage}"
        )
    print(jsonable(result.final))


if __name__ == "__main__":
    main()
