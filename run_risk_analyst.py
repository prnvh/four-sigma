from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents import (
    CompanyAnalysis,
    CompanyAnalysisRecord,
    Direction,
    OpenAIModelClient,
    OutcomeDefinition,
    RiskAnalyst,
)
from agents.run_company_analyst import load_context
from memory import ContextGateway
from memory.types import jsonable


def load_risk_store(path: Path):
    store = load_context(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("company_analyses", []):
        analysis = item["analysis"]
        store.append_company_analysis(
            CompanyAnalysisRecord(
                ref=str(item["ref"]),
                knowledge_time=datetime.fromisoformat(item["knowledge_time"]),
                analysis=CompanyAnalysis(
                    agent=str(analysis.get("agent", "company_analyst:v1")),
                    symbol=str(analysis["symbol"]), thesis=str(analysis["thesis"]),
                    fundamental_direction=Direction(analysis["fundamental_direction"]),
                    fundamental_confidence=float(analysis["fundamental_confidence"]),
                    momentum_direction=Direction(analysis["momentum_direction"]),
                    momentum_score=float(analysis["momentum_score"]),
                    momentum_confidence=float(analysis["momentum_confidence"]),
                    momentum_horizon=str(analysis["momentum_horizon"]),
                    evidence_refs=tuple(analysis["evidence_refs"]),
                    strengths=tuple(analysis["strengths"]), weaknesses=tuple(analysis["weaknesses"]),
                    catalysts=tuple(analysis["catalysts"]),
                    momentum_drivers=tuple(analysis["momentum_drivers"]),
                    momentum_risks=tuple(analysis["momentum_risks"]),
                    invalidation_conditions=tuple(analysis["invalidation_conditions"]),
                ),
            )
        )
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QFIRM AI Risk Analyst")
    parser.add_argument("symbol", help="Ticker being analysed")
    parser.add_argument("context", type=Path, help="JSON evidence and approved analyses")
    parser.add_argument("--horizon-days", type=int, required=True)
    parser.add_argument("--success-return-pct", type=float, required=True)
    parser.add_argument("--failure-return-pct", type=float, reuired=True)
    parser.add_argument("--as-of", type=datetime.fromisoformat)
    args = parser.parse_args()
    context = ContextGateway(load_risk_store(args.context)).for_risk_analyst(
        agent_id="risk_analyst",
        symbol=args.symbol,
        simulation_time=args.as_of or datetime.now(timezone.utc),
        outcome=OutcomeDefinition(
            horizon_days=args.horizon_days,
            success_return_pct=args.success_return_pct,
            failure_return_pct=args.failure_return_pct,
        ),
    )
    result = RiskAnalyst(OpenAIModelClient()).analyze(context)
    print(json.dumps(jsonable(result), indent=2))


if __name__ == "__main__":
    main()
