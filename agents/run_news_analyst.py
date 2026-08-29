from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from memory import ContextGateway, NewsEventStore

from .news_analyst import NewsAnalyst
from .model import OpenAIModelClient
from memory.types import Evidence, jsonable


def load_articles(path: Path) -> list[Evidence]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("article file must contain a JSON array")
    return [
        Evidence(
            ref=str(item["ref"]), source=str(item["source"]), url=str(item["url"]),
            published_at=datetime.fromisoformat(str(item["published_at"])),
            title=str(item["title"]), summary=str(item.get("summary", "")),
            symbols=tuple(item["symbols"]),
            knowledge_time=datetime.fromisoformat(str(item.get("knowledge_time", item["published_at"]))),
        )
        for item in payload
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QFIRM News Analyst")
    parser.add_argument("symbol", help="Ticker being analysed")
    parser.add_argument("articles", type=Path, help="JSON file containing sourced articles")
    parser.add_argument(
        "--as-of", type=datetime.fromisoformat,
        help="Timezone-aware simulation timestamp; defaults to current UTC time",
    )
    args = parser.parse_args()
    shared_memory = NewsEventStore()
    for article in load_articles(args.articles):
        shared_memory.append_news(article)
    context = ContextGateway(shared_memory).for_news_analyst(
        agent_id="news_analyst",
        symbol=args.symbol,
        simulation_time=args.as_of or datetime.now(timezone.utc),
    )
    finding = NewsAnalyst(OpenAIModelClient()).analyze(context)
    print(json.dumps(jsonable(finding), indent=2))


if __name__ == "__main__":
    main()
