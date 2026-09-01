"""Shared contracts for the next fine-tuning experiment."""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judge import TUTOR_SYSTEM_PROMPT

CANONICAL_SYSTEM_PROMPT = TUTOR_SYSTEM_PROMPT
CANONICAL_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    CANONICAL_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
MIN_TURNS = 5


def canonicalize_turns(turns: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return turns with the one approved system prompt and unchanged content."""
    result = [dict(turn) for turn in turns]
    system_turns = [turn for turn in result if turn.get("role") == "system"]
    if len(system_turns) != 1:
        raise ValueError("each training dialogue must contain exactly one system turn")
    system_turns[0]["content"] = CANONICAL_SYSTEM_PROMPT
    return result


def validate_training_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the exact conversation contract used by SFT."""
    validated: list[dict[str, Any]] = []
    for row in rows:
        turns = row.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"{row.get('id', '<unknown>')}: turns must be a list")
        roles = tuple(turn.get("role") for turn in turns)
        if len(roles) < MIN_TURNS or roles[0] != "system" or any(
            role != ("user" if index % 2 else "assistant")
            for index, role in enumerate(roles[1:], 1)
        ):
            raise ValueError(f"{row.get('id', '<unknown>')}: unexpected turn roles")
        if turns[0].get("content") != CANONICAL_SYSTEM_PROMPT:
            raise ValueError(f"{row.get('id', '<unknown>')}: non-canonical system prompt")
        if any(not isinstance(turn.get("content"), str) or not turn["content"].strip() for turn in turns):
            raise ValueError(f"{row.get('id', '<unknown>')}: empty message content")
        validated.append(dict(row))
    if not validated:
        raise ValueError("training dataset must not be empty")
    return validated


def rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash the exact canonicalized rows that enter the trainer."""
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
