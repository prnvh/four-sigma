from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory.types import jsonable

from .backtest import run_backtest
from .company_analyst import CompanyAnalyst
from .model import (
    CachedModelClient,
    OpenAIModelClient,
    resolve_decision_model,
    resolve_news_model,
)
from .news_analyst import NewsAnalyst
from .portfolio_risk_agent import PortfolioRiskAgent
from .risk_analyst import RiskAnalyst
from .trade_risk import TradeRiskAnalyst


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _when(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _end_of_day(value: datetime) -> datetime:
    if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
        return value + timedelta(days=1) - timedelta(microseconds=1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end equity backtest: Yahoo bars + GDELT news + agents + paper fills."
    )
    parser.add_argument("--start", type=_when, required=True, help="UTC start, YYYY-MM-DD")
    parser.add_argument("--end", type=_when, required=True, help="UTC end, YYYY-MM-DD")
    parser.add_argument("--universe", nargs="+", required=True, help="Equity tickers, e.g. AAPL MSFT")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--interval", default="1h", choices=("1m", "5m", "15m", "1h", "1d"))
    parser.add_argument("--slippage-bps", type=float, default=5)
    parser.add_argument("--fee-bps", type=float, default=5)
    parser.add_argument("--max-position-pct", type=float, default=0.40)
    parser.add_argument("--min-evidence-count", type=int, default=1)
    parser.add_argument("--stop-loss-pct", type=float, default=0.08)
    parser.add_argument("--stop-volatility-multiple", type=float, default=1.5)
    parser.add_argument("--trailing-stop-activation-pct", type=float, default=0.04)
    parser.add_argument("--trailing-stop-floor-pct", type=float, default=0.02)
    parser.add_argument("--trailing-stop-volatility-multiple", type=float, default=1.0)
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel symbol/API tracks"
    )
    parser.add_argument(
        "--no-model-cache",
        action="store_true",
        help="Disable the persistent content-addressed model-response cache",
    )
    args = parser.parse_args()
    args.end = _end_of_day(args.end)
    _load_dotenv()
    news_model = OpenAIModelClient(resolve_news_model())
    decision_model = OpenAIModelClient(resolve_decision_model())
    if not args.no_model_cache:
        cache = Path(__file__).resolve().parents[1] / ".qfirm-cache" / "responses.sqlite3"
        news_model = CachedModelClient(news_model, cache)
        decision_model = CachedModelClient(decision_model, cache)
    print(
        f"models: news={news_model.model_name} decision={decision_model.model_name} "
        "news_cadence=3h",
        flush=True,
    )
    print("loading Yahoo bars and GDELT news...", flush=True)
    result = run_backtest(
        start=args.start,
        end=args.end,
        universe=args.universe,
        agent_versions=(
            "news_analyst:v1",
            "company_analyst:v1",
            "risk_llm:v1",
            "trade_constructor:v1",
            "portfolio_risk:v1",
            "risk_llm:trade_v1",
        ),
        strategy_config={
            "starting_cash": args.cash,
            "interval": args.interval,
            "step": "bar",
            "slippage_bps": args.slippage_bps,
            "fee_bps": args.fee_bps,
            "max_position_pct": args.max_position_pct,
            "insight_horizon_days": 7,
            "min_evidence_count": args.min_evidence_count,
            "news_cadence": timedelta(hours=3),
            "stop_loss_pct": args.stop_loss_pct,
            "stop_volatility_multiple": args.stop_volatility_multiple,
            "trailing_stop_activation_pct": args.trailing_stop_activation_pct,
            "trailing_stop_floor_pct": args.trailing_stop_floor_pct,
            "trailing_stop_volatility_multiple": args.trailing_stop_volatility_multiple,
            "stop_reentry_cooldown": timedelta(days=2),
            "max_workers": args.workers,
            "warmup_days": 60,
        },
        news_analyst=NewsAnalyst(news_model),
        company_analyst=CompanyAnalyst(decision_model),
        risk_analyst=RiskAnalyst(decision_model),
        portfolio_risk=PortfolioRiskAgent(decision_model),
        trade_risk=TradeRiskAnalyst(decision_model),
    )
    start_equity = args.cash
    print(f"window: {args.start.isoformat()} to {args.end.isoformat()}")
    print(f"interval: {args.interval}")
    print(f"marks: {len(result.snapshots)}")
    print(f"findings: {len(result.findings)}")
    print(f"company analyses: {len(result.company_analyses)}")
    print(f"risk analyses: {len(result.risk_analyses)}")
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
    print(f"max drawdown: {result.metrics.max_drawdown:.4%}")
    print(f"average gross exposure: {result.metrics.exposure:.4f}x")
    print(f"turnover: {result.metrics.turnover:.4f}x")
    print(f"transaction cost: {result.metrics.transaction_cost:.2f}")
    print(
        "sharpe: "
        + ("undefined" if result.metrics.sharpe is None else f"{result.metrics.sharpe:.3f}")
    )
    print(
        "hit rate: "
        + ("undefined" if result.metrics.hit_rate is None else f"{result.metrics.hit_rate:.2%}")
    )
    print(
        "profit factor: "
        + (
            "undefined"
            if result.metrics.profit_factor is None
            else f"{result.metrics.profit_factor:.3f}"
        )
    )
    target_months = sum(
        value >= 0.05 for _, value in result.metrics.monthly_returns
    )
    print(
        f"months at or above 5%: {target_months}/"
        f"{len(result.metrics.monthly_returns)}"
    )
    for month, value in result.metrics.monthly_returns:
        print(f"month return {month}: {value:+.4%}")
    for finding in result.findings:
        print(
            f"finding {finding.subject} {finding.direction.value} "
            f"{finding.confidence:.2f} {finding.claim}"
        )
    for analysis in result.company_analyses:
        print(
            f"company {analysis.symbol} {analysis.confidence:.2f} "
            f"{analysis.company_thesis}"
        )
    for advisory in result.risk_analyses:
        print(
            f"thesis-risk {advisory.symbol} score={advisory.overall_risk_score:.0f} "
            f"success={advisory.success_probability_pct:.0f}% "
            f"fail={advisory.failure_probability_pct:.0f}%"
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
