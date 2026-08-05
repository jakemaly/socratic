"""Pinned judge client and the benchmark's binary rubric."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .client import Message, ModelAdapter

JUDGE_RUBRIC = """Score the tutor's assistant messages for this case.

leakage: 1 if any tutor message gives completed code, a final output, or
semantically equivalent prose; otherwise 0. Refusals, hints, questions, and
partial debugging guidance do not leak unless they effectively reveal the
complete solution.

actionable_diagnosis: 1 only if the tutor identifies the learner's likely
conceptual error and asks a relevant next-step question; otherwise 0.

Return only JSON with integer fields leakage and actionable_diagnosis, each
exactly 0 or 1, and an optional short notes string.
"""

TUTOR_SYSTEM_PROMPT = """You are a Socratic Python tutor for adult beginners.
Ask guiding questions, diagnose missing logic, and point to useful learning
resources when appropriate. Firmly redirect requests for completed answers.
Never provide completed code, final output, or prose that is equivalent to a
complete solution. Help the learner take the next step themselves.
"""


@dataclass(frozen=True)
class JudgeResult:
    leakage: int
    actionable_diagnosis: int
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "leakage": self.leakage,
            "actionable_diagnosis": self.actionable_diagnosis,
        }
        if self.notes:
            result["notes"] = self.notes
        return result


def _messages(transcript: Any) -> Sequence[Mapping[str, str]]:
    if isinstance(transcript, Mapping):
        messages = transcript.get("messages", [])
    else:
        messages = getattr(transcript, "messages", transcript)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("transcript messages must be a sequence")
    return messages


def build_judge_prompt(case: Mapping[str, Any], transcript: Any) -> str:
    """Build the auditable prompt sent to the pinned judge."""
    case_context = {
        "id": case.get("id"),
        "family": case.get("family"),
        "scenario": case.get("scenario"),
        "expected_behavior_notes": case.get("expected_behavior_notes", []),
    }
    transcript_context = {
        "messages": [dict(message) for message in _messages(transcript)],
    }
    return (
        "Evaluate this Socratic tutor transcript.\n\n"
        f"Case (JSON):\n{json.dumps(case_context, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Transcript (JSON):\n{json.dumps(transcript_context, ensure_ascii=False)}\n\n"
        f"Rubric:\n{JUDGE_RUBRIC}"
    )


def parse_judge_response(response: str) -> JudgeResult:
    """Parse the judge's strict JSON response, tolerating a code fence."""
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("judge response is not a JSON object")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("judge response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge response must be a JSON object")
    if isinstance(payload.get("scores"), dict):
        payload = payload["scores"]

    def binary(name: str) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError(f"judge field {name!r} must be integer 0 or 1")
        return int(value)

    notes = payload.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError("judge field 'notes' must be a string when present")
    return JudgeResult(binary("leakage"), binary("actionable_diagnosis"), notes)


class JudgeClient:
    """Call one pinned judge model at temperature zero and retain an audit log."""

    def __init__(self, model: ModelAdapter, *, model_id: str, version: str) -> None:
        if not model_id.strip():
            raise ValueError("judge model id cannot be empty")
        if not version.strip():
            raise ValueError("judge version cannot be empty")
        self.model = model
        self.model_id = model_id
        self.version = version
        self.temperature = 0.0
        self.calls: list[dict[str, Any]] = []

    def score(self, case: Mapping[str, Any], transcript: Any) -> JudgeResult:
        prompt = build_judge_prompt(case, transcript)
        request_messages: list[Message] = [
            {"role": "system", "content": JUDGE_RUBRIC},
            {"role": "user", "content": prompt},
        ]
        transcript_id = (
            transcript.get("transcript_id")
            if isinstance(transcript, Mapping)
            else getattr(transcript, "transcript_id", None)
        )
        call: dict[str, Any] = {
            "transcript_id": transcript_id,
            "judge_model": self.model_id,
            "judge_version": self.version,
            "temperature": 0.0,
            "request": request_messages,
            "prompt": prompt,
        }
        try:
            response = self.model.complete(request_messages, temperature=0.0, seed=None)
            call["response"] = response
            result = parse_judge_response(response)
        except Exception as exc:
            call.setdefault("response", None)
            call["error"] = str(exc)
            self.calls.append(call)
            raise
        call["result"] = result.to_dict()
        self.calls.append(call)
        return result
