from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from memory import ContextGateway, ResearchContextStore
from memory.types import (
    CompanyEntityRecord,
    CompanyRecord,
    CompanyRecordType,
    Direction,
    Evidence,
    MarketFeature,
    MarketFeatureType,
    PromotedInsight,
    jsonable,
)

from .company_analyst import CompanyAnalyst
from .model import OpenAIModelClient


def load_context(path: Path) -> ResearchContextStore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input file must contain a JSON object")
    memory = ResearchContextStore()
    for item in payload.get("company_records", []):
        memory.append_company_record(
            CompanyRecord(
                ref=str(item["ref"]), symbol=str(item["symbol"]), source=str(item["source"]),
                url=str(item["url"]), knowledge_time=datetime.fromisoformat(item["knowledge_time"]),
                record_type=CompanyRecordType(item["record_type"]), label=str(item["label"]),
                value=str(item["value"]), period_end=item.get("period_end"),
            )
        )
    company = payload.get("company")
    if company is not None:
        memory.append_company_entity(
            CompanyEntityRecord(
                ticker=str(company["ticker"]),
                exchange=str(company["exchange"]),
                sector=str(company["sector"]),
                industry=str(company["industry"]),
                identifiers=dict(company["identifiers"]),
                fundamental_references=tuple(company["fundamental_references"]),
                knowledge_time=datetime.fromisoformat(company["knowledge_time"]),
            )
        )
    for item in payload.get("recent_events", []):
        memory.append_news(
            Evidence(
                ref=str(item["ref"]),
                source=str(item["source"]),
                url=str(item["url"]),
                published_at=datetime.fromisoformat(item["published_at"]),
                title=str(item["title"]),
                summary=str(item.get("summary", "")),
                symbols=tuple(item.get("symbols", ())),
                knowledge_time=datetime.fromisoformat(item["knowledge_time"]),
            )
        )
    for item in payload.get("promoted_insights", []):
        valid_until = item.get("valid_until")
        memory.append_promoted_insight(
            PromotedInsight(
                ref=str(item["ref"]), symbol=str(item["symbol"]), claim=str(item["claim"]),
                direction=Direction(item["direction"]), confidence=float(item["confidence"]),
                evidence_refs=tuple(item["evidence_refs"]),
                knowledge_time=datetime.fromisoformat(item["knowledge_time"]),
                valid_until=datetime.fromisoformat(valid_until) if valid_until else None,
            )
        )
    for item in payload.get("market_features", []):
        memory.append_market_feature(
            MarketFeature(
                ref=str(item["ref"]), symbol=str(item["symbol"]), source=str(item["source"]),
                url=str(item["url"]), knowledge_time=datetime.fromisoformat(item["knowledge_time"]),
                feature=MarketFeatureType(item["feature"]), value=float(item["value"]),
                unit=str(item["unit"]),
            )
        )
    return memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QFIRM Company Analyst")
    parser.add_argument("symbol", help="Ticker being analysed")
    parser.add_argument("context", type=Path, help="JSON company evidence and promoted insights")
    parser.add_argument(
        "--as-of", type=datetime.fromisoformat,
        help="Timezone-aware simulation timestamp; defaults to current UTC time",
    )
    args = parser.parse_args()
    context = ContextGateway(load_context(args.context)).for_company_analyst(
        agent_id="company_analyst", symbol=args.symbol,
        simulation_time=args.as_of or datetime.now(timezone.utc),
    )
    analysis = CompanyAnalyst(OpenAIModelClient()).analyze(context)
    print(json.dumps(jsonable(analysis), indent=2))


if __name__ == "__main__":
    main()
