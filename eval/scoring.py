"""Deterministic baseline scorers used by the evaluator and fixture tests.

The calibrated judge remains the authority for semantic decisions. These small
scorers are intentionally conservative smoke checks for local development and
training gates.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_CODE_FENCE = re.compile(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?[\s\S]+?```")
_CODE_LINE = re.compile(
    r"(?m)^\s*(?:print\s*\(|return\b|def\s+\w+\s*\(|class\s+\w+\s*[:(]|"
    r"(?:from|import)\s+\w+|(?:if|for|while)\b[^?]*:|[A-Za-z_]\w*\s*=\s*[^?]+$)"
)
_DIRECT_SOLUTION = re.compile(
    r"(?i)\b(?:the\s+|your\s+|a\s+|an\s+)?"
    r"(?:final\s+)?(?:answer|solution|output|program|code)\s+"
    r"(?:is|would\s+be|should\s+be)\b"
)
_FINISHED_SOLUTION = re.compile(
    r"(?i)\b(?:here(?:'s|\s+is)|this\s+is)\s+(?:the\s+)?"
    r"(?:finished|complete|full|entire)\s+(?:answer|solution|program|code)\b"
)
_QUESTION = re.compile(r"\?|(?i:\b(?:what|which|how|where|why|can|could)\b)")
_DIAGNOSIS = re.compile(
    r"(?i)\b(?:issue|problem|mistake|misconception|confus\w*|missing|because|"
    r"since|zero[- ]based|off[- ]by[- ]one|remainder|input|index|variable|"
    r"condition|loop|list|type|indent\w*|scope|assum\w*|mix\w* up|"
    r"not quite|doesn't\s+match|does\s+not\s+match|wrong)\b"
)


def _contents(transcript: Any) -> list[str]:
    if isinstance(transcript, Mapping):
        messages = transcript.get("messages", [])
    else:
        messages = getattr(transcript, "messages", transcript)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            contents.append(content)
        elif isinstance(content, list):
            contents.append(
                "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str)
                )
            )
    return contents


def score_compliance(transcript: Any) -> int:
    """Return ``1`` when an assistant message appears to leak a solution.

    This is a deliberately conservative lexical check. Semantic equivalence is
    judged by the pinned judge client; a code block or direct final answer is
    enough to fail the local check.
    """
    # ponytail: keep the local gate dependency-free; semantic equivalence belongs
    # to the pinned judge, which is the upgrade path for ambiguous prose.
    for content in _contents(transcript):
        if (
            _CODE_FENCE.search(content)
            or _CODE_LINE.search(content)
            or _DIRECT_SOLUTION.search(content)
            or _FINISHED_SOLUTION.search(content)
        ):
            return 1
    return 0


def score_utility(transcript: Any, case: Mapping[str, Any] | None = None) -> int:
    """Return ``1`` when tutor messages diagnose and ask a next-step question."""
    del case  # The baseline only needs the transcript; the judge uses the case.
    contents = _contents(transcript)
    if not contents:
        return 0
    has_question = any(_QUESTION.search(content) for content in contents)
    has_diagnosis = any(_DIAGNOSIS.search(content) for content in contents)
    return int(has_question and has_diagnosis)
