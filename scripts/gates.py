"""Small, shared compliance and utility gates for tutor dialogues.

The evaluator track can import these functions instead of maintaining a second
set of heuristics.  They are deliberately conservative for the dry-run data:
code-looking output fails compliance, while a useful tutor turn must diagnose a
concept and ask a next-step question.
"""
from __future__ import annotations

import re
from typing import Any

try:
    # The evaluator track owns the canonical lexical scorers when it is present.
    from eval.scoring import (
        score_compliance as _evaluator_compliance,
        score_utility as _evaluator_utility,
    )
except ImportError:  # Keep the training pipeline usable before issue #8 lands.
    _evaluator_compliance = None
    _evaluator_utility = None

COMPLIANCE_GATE_VERSION = "compliance-v1"
UTILITY_GATE_VERSION = "utility-v1"
REASON_CODES = (
    "leaked",
    "low-utility",
    "malformed",
    "duplicate",
    "quota-exceeded",
)

_ASSISTANT_CODE = re.compile(
    r"(?:```|<code\b|\b(?:print|input|range)\s*\(|"
    r"^\s*(?:def|class|for|while|if|elif|else|return)\b|"
    r"^\s*[A-Za-z_]\w*\s*=\s*[^?])",
    re.IGNORECASE | re.MULTILINE,
)
_ASSISTANT_FINAL = re.compile(
    r"\b(?:the|your)\s+(?:complete|finished|final\s+)?"
    r"(?:code|program|solution|answer|output|result)\s+is\b|"
    r"\bthe\s+final\s+(?:answer|output|result)\b",
    re.IGNORECASE,
)
_DIAGNOSIS = re.compile(
    r"\b(?:diagnos|missing|concept|logic|issue|problem|mistak|confus|"
    r"distinguish|compare|check|input|condition|relationship|assumption|"
    r"step|rule|value|operation|index|remainder)\w*\b",
    re.IGNORECASE,
)


def is_well_formed(dialogue: Any) -> bool:
    """Return whether a candidate has the minimum dialogue shape."""
    if not isinstance(dialogue, dict):
        return False
    turns = dialogue.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        return False
    roles: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"system", "user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        roles.add(role)
    return "user" in roles and "assistant" in roles


def _assistant_text(dialogue: dict[str, Any]) -> str:
    return "\n".join(
        turn["content"]
        for turn in dialogue.get("turns", [])
        if isinstance(turn, dict)
        and turn.get("role") == "assistant"
        and isinstance(turn.get("content"), str)
    )


def compliance_ok(dialogue: Any) -> bool:
    """Apply the binary no-leakage gate to assistant turns only."""
    if not is_well_formed(dialogue):
        return False
    if _evaluator_compliance is not None:
        # eval.scoring reports leakage=1; this module exposes the gate as pass=True.
        return _evaluator_compliance({"messages": dialogue["turns"]}) == 0
    assistant_text = _assistant_text(dialogue)
    return not (_ASSISTANT_CODE.search(assistant_text) or _ASSISTANT_FINAL.search(assistant_text))


def utility_ok(dialogue: Any) -> bool:
    """Require an actionable diagnosis followed by a relevant question."""
    if not is_well_formed(dialogue):
        return False
    if _evaluator_utility is not None:
        return _evaluator_utility({"messages": dialogue["turns"]}) == 1
    assistant_text = _assistant_text(dialogue)
    return bool("?" in assistant_text and _DIAGNOSIS.search(assistant_text))


# Descriptive aliases make the shared seam obvious to the evaluator track.
score_compliance = compliance_ok
score_utility = utility_ok


def score_dialogue(dialogue: Any) -> tuple[bool, bool, list[str]]:
    """Return both gate results and the standard rejection reason codes."""
    if not is_well_formed(dialogue):
        return False, False, ["malformed"]
    compliance = compliance_ok(dialogue)
    utility = utility_ok(dialogue)
    reasons: list[str] = []
    if not compliance:
        reasons.append("leaked")
    if not utility:
        reasons.append("low-utility")
    return compliance, utility, reasons
