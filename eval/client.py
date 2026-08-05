"""Model adapters with a standard-library-only OpenAI-compatible client."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .scoring import score_compliance, score_utility

Message = dict[str, str]


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str: ...


class ModelError(RuntimeError):
    """Raised when a model adapter cannot produce a text completion."""


class OpenAICompatibleModel:
    """Call a ``/chat/completions`` endpoint without an SDK dependency."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key or os.getenv("SOCRATIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("SOCRATIC_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str:
        if not self.api_key:
            raise ModelError(
                "no API key; set SOCRATIC_API_KEY or OPENAI_API_KEY, "
                "or use a fixture model"
            )
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelError(f"model request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ModelError(f"model request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ModelError("model returned invalid JSON") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError("model response has no choices[0].message.content") from exc
        if isinstance(content, list):
            # Some compatible endpoints return structured text parts.
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            )
        if not isinstance(content, str):
            raise ModelError("model response content is not text")
        return content


class FixtureModel:
    """Deterministic local model for smoke tests and offline development."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.leaking = "leak" in model_id.lower()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str:
        del temperature, seed
        if self.leaking:
            return "Here is the finished program:\n```python\nprint(answer)\n```"
        latest_user = next(
            (
                message["content"]
                for message in reversed(messages)
                if message.get("role") == "user" and isinstance(message.get("content"), str)
            ),
            "the exercise",
        )
        return (
            f"The issue may be in the part of {latest_user.rstrip('?')} you are unsure about. "
            "What have you tried so far?"
        )


class HeuristicJudgeModel:
    """Offline judge adapter used only for ``fixture-judge`` runs."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str:
        del temperature, seed
        prompt = messages[-1].get("content", "") if messages else ""
        marker = "Transcript (JSON):\n"
        try:
            encoded = prompt.split(marker, 1)[1].split("\n\nRubric:", 1)[0]
            transcript = json.loads(encoded)
        except (IndexError, json.JSONDecodeError) as exc:
            raise ModelError("fixture judge could not read transcript prompt") from exc
        return json.dumps(
            {
                "leakage": score_compliance(transcript),
                "actionable_diagnosis": score_utility(transcript),
                "notes": "offline heuristic judge",
            }
        )


def make_model(model_id: str, *, base_url: str | None = None) -> ModelAdapter:
    """Build an offline fixture or OpenAI-compatible adapter from a model id."""
    if model_id.lower().startswith("fixture"):
        return FixtureModel(model_id)
    return OpenAICompatibleModel(model_id, base_url=base_url)


def make_judge_model(model_id: str, *, base_url: str | None = None) -> ModelAdapter:
    if model_id.lower().startswith("fixture"):
        return HeuristicJudgeModel()
    return OpenAICompatibleModel(model_id, base_url=base_url)
