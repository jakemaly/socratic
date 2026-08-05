"""Run benchmark cases and assemble the versioned EvalResult record."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .client import ModelAdapter
from .judge import JudgeClient, TUTOR_SYSTEM_PROMPT


@dataclass
class Transcript:
    transcript_id: str
    case_id: str
    model_id: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "transcript_id": self.transcript_id,
            "case_id": self.case_id,
            "model_id": self.model_id,
            "messages": [dict(message) for message in self.messages],
            "temperature": self.temperature,
            "seed": self.seed,
        }


@dataclass
class Evaluation:
    report: dict[str, Any]
    transcripts: list[Transcript]
    judge_calls: list[dict[str, Any]]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _rate(total: int, positive: int) -> float:
    return round(positive / total, 6) if total else 0.0


def _aggregates(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = (
        "normal-stuck",
        "answer-demand",
        "persistent-pressure",
        "misconception-edge",
    )
    by_family: dict[str, dict[str, Any]] = {}
    for family in families:
        family_scores = [score for score in scores if score.get("family") == family]
        by_family[family] = {
            "count": len(family_scores),
            "leakage_rate": _rate(
                len(family_scores),
                sum(score.get("leakage", 0) for score in family_scores),
            ),
            "diagnosis_rate": _rate(
                len(family_scores),
                sum(score.get("actionable_diagnosis", 0) for score in family_scores),
            ),
        }
    return {
        "case_count": len(scores),
        "leakage_rate": _rate(len(scores), sum(score["leakage"] for score in scores)),
        "diagnosis_rate": _rate(
            len(scores), sum(score["actionable_diagnosis"] for score in scores)
        ),
        "by_family": by_family,
    }


def evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    model: ModelAdapter,
    *,
    model_id: str,
    judge: JudgeClient,
    run_id: str | None = None,
    adapter_id: str | None = None,
    temperature: float = 0.0,
    seed: int = 0,
) -> Evaluation:
    """Run each learner turn through one accumulating chat transcript."""
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not cases:
        raise ValueError("at least one benchmark case is required")
    effective_run_id = run_id or _run_id()
    transcripts: list[Transcript] = []
    scores: list[dict[str, Any]] = []

    for case in cases:
        case_id = case.get("id")
        family = case.get("family")
        learner_turns = case.get("learner_turns")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each benchmark case needs a non-empty id")
        if not isinstance(family, str) or not family:
            raise ValueError(f"case {case_id!r} needs a family")
        if not isinstance(learner_turns, list) or not learner_turns:
            raise ValueError(f"case {case_id!r} needs learner_turns")
        if any(not isinstance(turn, str) or not turn for turn in learner_turns):
            raise ValueError(f"case {case_id!r} has an invalid learner turn")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": TUTOR_SYSTEM_PROMPT}
        ]
        for learner_turn in learner_turns:
            messages.append({"role": "user", "content": learner_turn})
            response = model.complete(messages, temperature=temperature, seed=seed)
            if not isinstance(response, str):
                raise TypeError(f"model response for {case_id!r} is not text")
            messages.append({"role": "assistant", "content": response})

        transcript = Transcript(
            transcript_id=f"{effective_run_id}-{case_id}",
            case_id=case_id,
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )
        transcripts.append(transcript)
        judged = judge.score(case, transcript)
        score: dict[str, Any] = {
            "case_id": case_id,
            "family": family,
            "leakage": judged.leakage,
            "actionable_diagnosis": judged.actionable_diagnosis,
        }
        if judged.notes:
            score["notes"] = judged.notes
        scores.append(score)

    report = {
        "schema_version": "1.0.0",
        "run_id": effective_run_id,
        "model_id": model_id,
        "adapter_id": adapter_id,
        "judge_version": judge.version,
        "temperature": temperature,
        "seed": seed,
        "case_scores": scores,
        "aggregates": _aggregates(scores),
    }
    return Evaluation(report, transcripts, list(judge.calls))
