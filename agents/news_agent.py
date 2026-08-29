from __future__ import annotations

import re
from typing import Any

from memory.capabilities import Action, CAPABILITIES
from memory.context_gateway import NewsAgentContext
from memory.types import NewsCategory, NewsObservation, jsonable

from .model import ModelClient
from .registry import NEWS_V1, AgentSpec


NEWS_OBSERVATIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "event_id": {"type": "string"},
                    "entities": {
                        "type": "array", "items": {"type": "string"}, "minItems": 1,
                    },
                    "category": {
                        "type": "string", "enum": [item.value for item in NewsCategory],
                    },
                    "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["event_id", "entities", "category", "relevance"],
            },
        }
    },
    "required": ["observations"],
}


class NewsAgent:
    """Deduplicates and classifies news without investment conclusions."""

    def __init__(self, model: ModelClient, spec: AgentSpec = NEWS_V1) -> None:
        if spec.name != "news":
            raise ValueError("NewsAgent requires a news spec")
        self.model = model
        self.spec = spec

    def observe(self, context: NewsAgentContext) -> tuple[NewsObservation, ...]:
        if not isinstance(context, NewsAgentContext):
            raise TypeError("NewsAgent requires context from ContextGateway")
        CAPABILITIES.require("news", Action.READ, "events", "news")
        articles = _deduplicate(context.articles)
        if not articles:
            return ()
        allowed_refs = {item.ref: item for item in articles}
        universe = set(context.universe)
        result = self.model.generate_json(
            instructions=self.spec.prompt,
            input_data={
                "simulation_time": context.simulation_time.isoformat(),
                "universe": list(context.universe),
                "articles": [jsonable(item) for item in articles],
            },
            schema=NEWS_OBSERVATIONS_SCHEMA,
        )
        if not isinstance(result, dict) or set(result) != {"observations"}:
            raise ValueError("news agent output must contain only observations")
        raw = result["observations"]
        if not isinstance(raw, list):
            raise ValueError("observations must be a list")
        observations: list[NewsObservation] = []
        seen: set[str] = set()
        for item in raw:
            required = {"event_id", "entities", "category", "relevance"}
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("news observation has missing or unexpected fields")
            event_id = item["event_id"]
            if not isinstance(event_id, str) or event_id not in allowed_refs:
                raise ValueError("news observation cited an unknown event")
            if event_id in seen:
                raise ValueError("news observations cannot duplicate an event")
            entities = item["entities"]
            if not isinstance(entities, list) or not entities or not all(
                isinstance(entity, str) and entity.strip() for entity in entities
            ):
                raise ValueError("news observation entities must be non-empty strings")
            normalized = tuple(entity.strip().upper() for entity in entities)
            if len(set(normalized)) != len(normalized) or set(normalized) - universe:
                raise ValueError("news observation contains duplicate or unknown entities")
            tagged = set(allowed_refs[event_id].symbols)
            if tagged and set(normalized) - tagged:
                raise ValueError("news observation contradicts supplied entity tags")
            try:
                category = NewsCategory(item["category"])
            except (TypeError, ValueError) as error:
                raise ValueError("news observation category is invalid") from error
            observations.append(
                NewsObservation(
                    event_id=event_id,
                    entities=normalized,
                    category=category,
                    relevance=item["relevance"],
                )
            )
            seen.add(event_id)
        return tuple(observations)


def _deduplicate(articles):
    selected = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for article in articles:
        url = article.url.strip().lower().rstrip("/")
        title = re.sub(r"[^a-z0-9]+", " ", article.title.lower()).strip()
        if url in seen_urls or title in seen_titles:
            continue
        seen_urls.add(url)
        seen_titles.add(title)
        selected.append(article)
    return tuple(selected)
