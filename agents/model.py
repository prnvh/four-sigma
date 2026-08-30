from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


class ModelClient(Protocol):
    def generate_json(
        self, *, instructions: str, input_data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]: ...


NEWS_MODEL = "gpt-4.1-nano"
DECISION_MODEL = "gpt-4.1-mini"


def resolve_news_model() -> str:
    return (os.getenv("QFIRM_NEWS_MODEL") or NEWS_MODEL).strip()


def resolve_decision_model() -> str:
    return (os.getenv("QFIRM_DECISION_MODEL") or DECISION_MODEL).strip()


class OpenAIModelClient:
    """Small Responses API adapter; no finance logic lives in this class."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install requirements.txt before using OpenAI") from exc
        self._model = model or resolve_decision_model()
        if not self._model.strip():
            raise RuntimeError("set QFIRM_NEWS_MODEL or QFIRM_DECISION_MODEL")
        self._client = OpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    def generate_json(
        self, *, instructions: str, input_data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, 8):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    instructions=instructions,
                    input=json.dumps(input_data, separators=(",", ":"), ensure_ascii=False),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "qfirm_agent_output",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    store=False,
                )
                return json.loads(response.output_text)
            except Exception as exc:
                last = exc
                name = type(exc).__name__
                retryable = (
                    "Connection" in name
                    or "Timeout" in name
                    or name == "RateLimitError"
                )
                if not retryable:
                    raise
                hinted = re.search(r"try again in ([\d.]+)s", str(exc))
                wait = float(hinted.group(1)) + 1 if hinted else min(5 * attempt, 20)
                print(f"OpenAI {name}, retry {attempt}/7 in {wait:.1f}s", flush=True)
                time.sleep(wait)
        raise RuntimeError("OpenAI request failed after retries") from last


class CachedModelClient:
    """Content-addressed, thread-safe cache for deterministic backtest replays."""

    def __init__(self, delegate: ModelClient, path: str | os.PathLike[str]) -> None:
        if not hasattr(delegate, "generate_json"):
            raise TypeError("delegate must implement generate_json")
        self._delegate = delegate
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        with closing(sqlite3.connect(self._path, timeout=30)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS responses "
                "(cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL)"
            )
            connection.commit()

    @property
    def model_name(self) -> str:
        return str(getattr(self._delegate, "model_name", type(self._delegate).__name__))

    def generate_json(
        self, *, instructions: str, input_data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "cache_version": 1,
                "model": self.model_name,
                "instructions": instructions,
                "input": input_data,
                "schema": schema,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        key = sha256(payload.encode("utf-8")).hexdigest()
        with self._lock:
            with closing(sqlite3.connect(self._path, timeout=30)) as connection:
                row = connection.execute(
                    "SELECT response_json FROM responses WHERE cache_key = ?", (key,)
                ).fetchone()
        if row is not None:
            return json.loads(row[0])
        result = self._delegate.generate_json(
            instructions=instructions, input_data=input_data, schema=schema
        )
        encoded = json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        with self._lock:
            with closing(sqlite3.connect(self._path, timeout=30)) as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO responses(cache_key, response_json) VALUES (?, ?)",
                    (key, encoded),
                )
                connection.commit()
        return json.loads(encoded)
