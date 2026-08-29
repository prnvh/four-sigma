from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol


class ModelClient(Protocol):
    def generate_json(
        self, *, instructions: str, input_data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]: ...


class OpenAIModelClient:
    """Small Responses API adapter; no finance logic lives in this class."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install requirements.txt before using OpenAI") from exc
        self._model = model or os.getenv("QFIRM_MODEL") or "gpt-4.1"
        if not self._model.strip():
            raise RuntimeError("set QFIRM_MODEL to a model available in your account")
        self._client = OpenAI()

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
