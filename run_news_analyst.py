from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from agents import Evidence, NewsAnalyst, OpenAIModelClient
from agents.schemas import jsonable


def load_articles(path: Path) -> list[Evidence]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("article file must contain a JSON array")
    return [
        Evidence(
            ref=str(item["ref"]), source=str(item["source"]), url=str(item["url"]),
            published_at=datetime.fromisoformat(str(item["published_at"])),
            title=str(item["title"]), summary=str(item.get("summary", "")),
        )
        for item in payload
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QFIRM News Analyst")
    parser.add_argument("symbol", help="Ticker being analysed")
    parser.add_argument("articles", type=Path, help="JSON file containing sourced articles")
    args = parser.parse_args()
    finding = NewsAnalyst(OpenAIModelClient()).analyze(args.symbol, load_articles(args.articles))
    print(json.dumps(jsonable(finding), indent=2))


if __name__ == "__main__":
    main()
