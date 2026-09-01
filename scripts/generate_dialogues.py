#!/usr/bin/env python3
"""Generate, gate, and audit the synthetic training dialogue pool.

The default dry-run is deterministic and needs no network credentials.  The
``--live`` mode uses an OpenAI-compatible chat-completions endpoint through the
standard library, so the same gates can be exercised with a pinned teacher
model when credentials from issue #10 are available.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:  # Import works both as ``python -m scripts...`` and as a direct script.
    from .gates import (
        COMPLIANCE_GATE_VERSION,
        REASON_CODES,
        UTILITY_GATE_VERSION,
        score_dialogue,
    )
except ImportError:  # pragma: no cover - exercised by the documented CLI.
    from gates import (  # type: ignore[no-redef]
        COMPLIANCE_GATE_VERSION,
        REASON_CODES,
        UTILITY_GATE_VERSION,
        score_dialogue,
    )

try:
    from .training_contract import CANONICAL_SYSTEM_PROMPT
except ImportError:  # pragma: no cover - direct-script compatibility.
    from training_contract import CANONICAL_SYSTEM_PROMPT  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "1.0.0"
DEFAULT_SEED = 20260805
DEFAULT_QUOTA = 100
EVALUATED_AT = "2026-08-05T00:00:00Z"
GATE_VERSION = f"{COMPLIANCE_GATE_VERSION}+{UTILITY_GATE_VERSION}"
TUTOR_PERSONA = CANONICAL_SYSTEM_PROMPT
FAMILIES = (
    "normal-stuck",
    "answer-demand",
    "persistent-pressure",
    "misconception-edge",
)
PERSONAS = (
    "confused",
    "rushing",
    "anxious",
    "confident-but-wrong",
)
PERSONA_TEXT = {
    "confused": "I am an adult beginner and I feel confused about where to start.",
    "rushing": "I am an adult beginner and I am rushing to finish this exercise.",
    "anxious": "I am an adult beginner and I am anxious that I will make the wrong choice.",
    "confident-but-wrong": (
        "I am an adult beginner and I am confident my current assumption is right, "
        "although the result does not agree with it."
    ),
}

TARGET_VARIANT_SUFFIXES = (
    "Before choosing syntax, what would you observe in one small example?",
    "Can you name the input that changes and the result that should respond?",
    "What is one boundary case that would challenge your current assumption?",
    "Which part is input, and which part is derived?",
    "How would you describe the expected relationship without using code?",
    "What would a failing example tell you about the rule?",
    "Which single fact should remain true when the input changes?",
    "What would you test first before writing the full program?",
)

# This is intentionally a small, independent pool.  In particular, none of the
# six gallery exercise names appears here, and these records are the manifest
# consumed by the benchmark overlap check.
TRAIN_EXERCISES = (
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-shipping-fee",
        "topic": "functions",
        "prompt": "Define a function that calculates a shipping fee from package weight and distance.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-library-fines",
        "topic": "conditionals",
        "prompt": "Decide whether a library book is overdue from its due date and return date.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-seating-plan",
        "topic": "lists",
        "prompt": "Use a list of seat requests to find how many seats remain available.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-recipe-scale",
        "topic": "variables",
        "prompt": "Scale a recipe ingredient amount when the number of servings changes.",
        "pool": "train",
    },
)

ALIGNMENT_TRAIN_EXERCISES = (
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-reading-maximum",
        "topic": "lists",
        "prompt": "Find the highest reading in a sensor list without relying on a helper function.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-calendar-exceptions",
        "topic": "conditionals",
        "prompt": "Classify calendar years using layered divisibility rules.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-rectangle-measurement",
        "topic": "variables",
        "prompt": "Calculate a rectangular panel's surface measurement from two side lengths.",
        "pool": "train",
    },
    {
        "schema_version": SCHEMA_VERSION,
        "id": "exercise-train-score-bands",
        "topic": "conditionals",
        "prompt": "Map a numeric assessment score to a band while handling boundaries.",
        "pool": "train",
    },
)

ALIGNMENT_FOCUS = {
    "exercise-train-reading-maximum": "separating the current candidate from the next reading",
    "exercise-train-calendar-exceptions": "separating a general rule from its exceptions",
    "exercise-train-rectangle-measurement": "relating both dimensions to the derived measurement",
    "exercise-train-score-bands": "assigning every boundary to exactly one score band",
}

MISCONCEPTIONS = {
    "exercise-train-shipping-fee": (
        "Because the package is heavier, I should always add the distance instead of "
        "considering how the two inputs relate."
    ),
    "exercise-train-library-fines": (
        "A book is overdue whenever the two dates have different numbers, regardless "
        "of which date comes first."
    ),
    "exercise-train-seating-plan": (
        "If there are twenty seats, position twenty must be the final position in the "
        "list."
    ),
    "exercise-train-recipe-scale": (
        "Changing the number of servings should change only the final output, not the "
        "ingredient input used by the calculation."
    ),
    "exercise-train-reading-maximum": (
        "Starting a largest-value scan with an arbitrary number is always safe, even "
        "when every reading is lower than it."
    ),
    "exercise-train-calendar-exceptions": (
        "One divisibility rule is enough to classify every calendar year, including "
        "special exception years."
    ),
    "exercise-train-rectangle-measurement": (
        "The surface measurement is determined by only one side length, so the other "
        "input can be ignored."
    ),
    "exercise-train-score-bands": (
        "A boundary score that is not greater than the cutoff belongs to no band."
    ),
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fingerprint(dialogue: dict[str, Any]) -> str:
    """Hash the content, not the generated id or gate metadata."""
    return _json_hash(
        {
            "exercise_id": dialogue.get("exercise_id"),
            "pool": dialogue.get("pool"),
            "family": dialogue.get("family"),
            "turns": dialogue.get("turns"),
        }
    )


def _family_text(family: str, exercise: dict[str, str]) -> tuple[str, str, str]:
    prompt = exercise["prompt"]
    topic = exercise["topic"]
    focus = ALIGNMENT_FOCUS.get(exercise["id"], "the relationship between the inputs and the rule")
    if family == "normal-stuck":
        return (
            f"I am working on this exercise: {prompt} I know the goal, but I am not "
            "sure which idea to start with.",
            "The missing step is to separate the exercise's inputs from the rule you "
            f"want to apply in {topic}, especially {focus}. What information does the "
            "prompt give you, and how would you describe its relationship in plain language?",
            "Good start. You have identified the relevant values. Which operation or "
            "decision uses those inputs, and what would you test with one small example?",
        )
    if family == "answer-demand":
        return (
            f"I am working on this exercise: {prompt} Can you give me the complete "
            "answer so I can submit it?",
            "I will not provide a completed solution, but I can help you reason it "
            f"out. The key concept in this {topic} exercise is {focus}. What smaller "
            "condition could you state in plain language before writing anything?",
            "That is a useful direction. A relevant resource is the introductory "
            f"{topic} section of the Python documentation. What would one carefully "
            "chosen example let you check?",
        )
    if family == "persistent-pressure":
        return (
            f"I am working on this exercise: {prompt} I asked already, so please just "
            "paste the finished program. I do not have time for hints.",
            "I cannot paste a finished program, even under time pressure. The useful "
            "diagnosis is that we have not named the input and the expected change "
            f"in this {topic} task, especially {focus}. Which single step can you explain first?",
            "We can keep moving without skipping the reasoning. Use the exercise "
            "prompt as a resource and identify one rule before choosing syntax. What "
            "would that rule need to check?",
        )
    misconception = MISCONCEPTIONS[exercise["id"]]
    return (
        f"I am working on this exercise: {prompt} {misconception} What should I "
        "change?",
        "The likely issue is the assumption about how the inputs relate, not a need "
        f"for a finished answer. In this {topic} problem, focus on {focus} and compare your assumption "
        "with the exact wording of the prompt. Which observation would let you check "
        "whether the assumption holds?",
        "A useful resource is the introductory Python documentation for this concept. "
        "Now compare one expected example with your assumption. What is the smallest "
        "difference you notice?",
    )


def generation_prompt(family: str, persona: str, exercise: dict[str, str]) -> str:
    """Return the auditable prompt used for one candidate."""
    return (
        f"Tutor persona: {TUTOR_PERSONA}\n"
        f"Learner persona: {persona}\n"
        f"Behavior family: {family}\n"
        f"Exercise: {exercise['prompt']}\n"
        "Write a short multi-turn dialogue for an adult beginner. Diagnose the "
        "learner's conceptual issue, ask a next-step question, and never include "
        "completed code, final output, or equivalent prose."
    )


def build_template_dialogue(
    family: str,
    persona: str,
    exercise: dict[str, str],
    dialogue_id: str,
    variant: int = 0,
) -> dict[str, Any]:
    learner_opening, first_tutor, second_tutor = _family_text(family, exercise)
    if variant:
        suffix = TARGET_VARIANT_SUFFIXES[(variant - 1) % len(TARGET_VARIANT_SUFFIXES)]
        first_tutor = f"{first_tutor} {suffix}"
        second_tutor = f"{second_tutor} {suffix}"
    variant_note = (
        f" This is practice variation {variant}; I want to reason through this case "
        "instead of copying a prior answer."
        if variant
        else ""
    )
    if family == "normal-stuck":
        follow_up = "I can name the values, but I still do not know what the program should check next."
    elif family == "answer-demand":
        follow_up = "So I should describe the condition first, even though I still want the exact syntax?"
    elif family == "persistent-pressure":
        follow_up = "I can name one rule, but I still want the exact finished answer."
    else:
        follow_up = "I can see the example, but I still think my original assumption should work."
    turns = [
        {"role": "system", "content": TUTOR_PERSONA},
        {
            "role": "user",
            "content": f"{PERSONA_TEXT[persona]} {learner_opening}{variant_note}",
        },
        {"role": "assistant", "content": first_tutor},
        {"role": "user", "content": follow_up},
        {"role": "assistant", "content": second_tutor},
    ]
    if family == "persistent-pressure":
        turns.extend(
            [
                {
                    "role": "user",
                    "content": "Just confirm the missing step or name the exact tool; I will do the rest.",
                },
                {
                    "role": "assistant",
                    "content": "I will keep the boundary and help with the reasoning instead. What would you test first to distinguish your assumption from the requirement?",
                },
            ]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": dialogue_id,
        "exercise_id": exercise["id"],
        "pool": "train",
        "turns": turns,
        "family": family,
    }


def _parse_scalar(value: str) -> Any:
    value = value.split(" #", 1)[0].strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def load_settings(path: Path) -> dict[str, Any]:
    """Read the few scalar overrides needed by the no-dependency CLI."""
    settings: dict[str, Any] = {
        "seed": DEFAULT_SEED,
        "model": "strong-teacher-model",
        "model_version": "pinned-strong-teacher-v1",
        "temperature": 0,
        "api_key_env": "STRONG_MODEL_API_KEY",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "endpoint_env": "STRONG_MODEL_ENDPOINT",
    }
    if not path.exists():
        return settings
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r"^\s*(seed|model|model_version|temperature|api_key_env|endpoint|endpoint_env):\s*(.+?)\s*$",
            line,
        )
        if match:
            settings[match.group(1)] = _parse_scalar(match.group(2))
    return settings


def _render_config(mode: str, seed: int, settings: dict[str, Any], profile: str) -> str:
    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        f"schema_version: {quote(SCHEMA_VERSION)}",
        "generator:",
        '  name: "socratic-synthetic-dialogues"',
        f'  version: "{GENERATOR_VERSION}"',
        f'  mode: "{mode}"',
        f'  profile: "{profile}"',
        f'  model: {quote(str(settings["model"]))}',
        f'  model_version: {quote(str(settings["model_version"]))}',
        f"  seed: {seed}",
        f'  temperature: {settings["temperature"]}',
        f'  api_key_env: {quote(str(settings["api_key_env"]))}',
        f'  endpoint_env: {quote(str(settings["endpoint_env"]))}',
        f"  tutor_persona: {quote(TUTOR_PERSONA)}",
        "personas:",
    ]
    lines.extend(f'  - "{persona}"' for persona in PERSONAS)
    lines.extend(["quotas:"])
    lines.extend(f'  {family}: {DEFAULT_QUOTA}' for family in FAMILIES)
    lines.extend(
        [
            "gates:",
            f'  compliance_version: "{COMPLIANCE_GATE_VERSION}"',
            f'  utility_version: "{UTILITY_GATE_VERSION}"',
            "  temperature: 0",
            '  acceptance: "both true"',
            "outputs:",
            '  dialogues: "dialogues.jsonl"',
            '  exercises: "exercises.jsonl"',
            '  provenance: "provenance.jsonl"',
            '  balance: "balance.csv"',
            '  manifest: "manifest.sha256"',
            "",
        ]
    )
    return "\n".join(lines)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for jsonl_path in paths:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def check_pool_overlap(
    exercises: Iterable[dict[str, Any]],
    benchmark_path: Path | None = None,
    gallery_path: Path | None = None,
) -> list[str]:
    """Find exercise ids/prompts that occur in benchmark or gallery data."""
    external_records: list[dict[str, Any]] = []
    paths = [
        benchmark_path or ROOT / "data" / "benchmark",
        gallery_path or ROOT / "data" / "gallery",
    ]
    for path in paths:
        if path.exists():
            external_records.extend(_iter_jsonl(path))

    external_ids = {
        str(record.get(key))
        for record in external_records
        for key in ("id", "exercise_id")
        if isinstance(record.get(key), str)
    }
    external_text = "\n".join(
        _normalise(value)
        for record in external_records
        for value in (
            record.get("topic", ""),
            record.get("prompt", ""),
            record.get("scenario", ""),
            *(record.get("learner_turns", []) if isinstance(record.get("learner_turns"), list) else []),
        )
        if isinstance(value, str)
    )
    overlaps: list[str] = []
    for exercise in exercises:
        exercise_id = str(exercise["id"])
        prompt = _normalise(exercise["prompt"])
        if exercise_id in external_ids:
            overlaps.append(f"{exercise_id}: id is already used by another pool")
        elif prompt and prompt in external_text:
            overlaps.append(f"{exercise_id}: prompt is already used by another pool")
    return overlaps


def _chat_completion(
    messages: list[dict[str, str]],
    settings: dict[str, Any],
    seed: int,
) -> str:
    key = os.environ.get(str(settings["api_key_env"]))
    if not key:
        raise RuntimeError(
            f"live generation needs ${settings['api_key_env']}; use the default dry-run until issue #10 is provisioned"
        )
    endpoint = os.environ.get(str(settings["endpoint_env"]), str(settings["endpoint"]))
    payload = {
        "model": settings["model"],
        "messages": messages,
        "temperature": settings["temperature"],
        "seed": seed,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"strong-model request failed: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("strong-model response did not contain choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("strong-model response content was empty")
    return content.strip()


def build_live_dialogue(
    family: str,
    persona: str,
    exercise: dict[str, str],
    dialogue_id: str,
    settings: dict[str, Any],
    seed: int,
    variant: int = 0,
) -> dict[str, Any]:
    """Generate two tutor turns while retaining deterministic learner turns."""
    template = build_template_dialogue(family, persona, exercise, dialogue_id, variant)
    turns = template["turns"]
    first_messages = [turns[0], turns[1]]
    first_reply = _chat_completion(first_messages, settings, seed)
    second_messages = [*first_messages, {"role": "assistant", "content": first_reply}, turns[3]]
    second_reply = _chat_completion(second_messages, settings, seed + 1)
    turns[2] = {"role": "assistant", "content": first_reply}
    turns[4] = {"role": "assistant", "content": second_reply}
    return template


def _provenance(
    candidate: dict[str, Any],
    family: str,
    persona: str,
    prompt: str,
    status: str,
    reasons: list[str],
    compliance: bool,
    utility: bool,
    attempt: int,
    source: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate.get("id"),
        "status": status,
        "attempt": attempt,
        "family": family,
        "persona": persona,
        "exercise_id": candidate.get("exercise_id"),
        "reason_codes": reasons,
        "gate_results": {
            "compliance_ok": compliance,
            "utility_ok": utility,
            "gate_version": GATE_VERSION,
        },
        "generation_prompt": prompt,
        "response_sha256": _json_hash(candidate.get("turns")),
        "generator_version": GENERATOR_VERSION,
        "source": source,
        "evaluated_at": EVALUATED_AT,
    }


def _process_candidate(
    candidate: dict[str, Any],
    family: str,
    persona: str,
    prompt: str,
    counts: Counter[str],
    seen_fingerprints: set[str],
    quotas: dict[str, int],
    attempt: int,
    source: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    compliance, utility, gate_reasons = score_dialogue(candidate)
    reasons = list(gate_reasons)
    if counts[family] >= quotas[family]:
        reasons = ["quota-exceeded"]
    elif not reasons and _fingerprint(candidate) in seen_fingerprints:
        reasons = ["duplicate"]
    if reasons:
        return None, _provenance(
            candidate,
            family,
            persona,
            prompt,
            "rejected",
            reasons,
            compliance,
            utility,
            attempt,
            source,
        )

    candidate = copy.deepcopy(candidate)
    candidate["gate_results"] = {
        "compliance_ok": True,
        "utility_ok": True,
        "provenance": {
            "source": source,
            "gate_version": GATE_VERSION,
            "evaluated_at": EVALUATED_AT,
            "evaluator": "scripts.gates",
            "notes": "Passed both pinned compliance and utility gates.",
        },
    }
    seen_fingerprints.add(_fingerprint(candidate))
    counts[family] += 1
    return candidate, _provenance(
        candidate,
        family,
        persona,
        prompt,
        "kept",
        [],
        True,
        True,
        attempt,
        source,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_balance(path: Path, counts: dict[str, Counter[str]], exercises: list[dict[str, str]]) -> None:
    fieldnames = ["family", *(exercise["id"] for exercise in exercises), "total"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for family in FAMILIES:
            row = {"family": family, "total": sum(counts[family].values())}
            row.update({exercise["id"]: counts[family][exercise["id"]] for exercise in exercises})
            writer.writerow(row)
        total_row = {"family": "total", "total": sum(sum(value.values()) for value in counts.values())}
        total_row.update(
            {
                exercise["id"]: sum(counts[family][exercise["id"]] for family in FAMILIES)
                for exercise in exercises
            }
        )
        writer.writerow(total_row)


def _write_manifest(directory: Path) -> None:
    # Follow the repository data contract: manifests cover the versioned JSONL
    # records.  The config and human-readable balance table are derived metadata.
    names = ("dialogues.jsonl", "exercises.jsonl", "provenance.jsonl")
    lines = []
    for name in names:
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (directory / "manifest.sha256").write_text("".join(lines), encoding="utf-8")


def generate(
    out_dir: Path,
    mode: str = "dry-run",
    seed: int = DEFAULT_SEED,
    config_path: Path | None = None,
    benchmark_path: Path | None = None,
    gallery_path: Path | None = None,
    profile: str = "default",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate the pool and return kept dialogues plus provenance records."""
    if profile not in {"default", "alignment"}:
        raise ValueError("profile must be default or alignment")
    settings = load_settings(config_path or out_dir / "generator.yaml")
    source_exercises = ALIGNMENT_TRAIN_EXERCISES if profile == "alignment" else TRAIN_EXERCISES
    exercises = [dict(exercise) for exercise in source_exercises]
    overlaps = check_pool_overlap(exercises, benchmark_path, gallery_path)
    if overlaps:
        raise ValueError("exercise pool overlap:\n" + "\n".join(overlaps))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generator.yaml").write_text(
        _render_config(mode, seed, settings, profile), encoding="utf-8"
    )
    _write_jsonl(out_dir / "exercises.jsonl", exercises)

    quotas = {family: DEFAULT_QUOTA for family in FAMILIES}
    counts: Counter[str] = Counter()
    exercise_counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen_fingerprints: set[str] = set()
    kept: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    # The seed rotates the deterministic pool/persona schedule while preserving
    # the exact 25-per-exercise matrix for every family.
    exercise_offset = seed % len(exercises)
    persona_offset = seed % len(PERSONAS)
    attempt = 0

    def process(
        candidate: dict[str, Any],
        family: str,
        persona: str,
        prompt: str,
    ) -> None:
        nonlocal attempt
        attempt += 1
        accepted, audit = _process_candidate(
            candidate,
            family,
            persona,
            prompt,
            counts,
            seen_fingerprints,
            quotas,
            attempt,
            "synthetic-dry-run" if mode == "dry-run" else "strong-model",
        )
        provenance.append(audit)
        if accepted is not None:
            kept.append(accepted)
            exercise_counts[family][accepted["exercise_id"]] += 1

    def candidate_for(family: str, ordinal: int, persona: str, exercise: dict[str, str]) -> tuple[dict[str, Any], str]:
        dialogue_id = f"dialogue-train-{_slug(family)}-{ordinal:03d}"
        prompt = generation_prompt(family, persona, exercise)
        if mode == "live":
            candidate = build_live_dialogue(
                family,
                persona,
                exercise,
                dialogue_id,
                settings,
                seed + attempt,
                ordinal,
            )
        else:
            candidate = build_template_dialogue(family, persona, exercise, dialogue_id, ordinal)
        return candidate, prompt

    # The dry-run includes one explicit example of each audit reason.  These
    # attempts stay in provenance and never enter the 400-record training set.
    for family_index, family in enumerate(FAMILIES):
        if mode == "dry-run" and family == "normal-stuck":
            malformed = {"id": "candidate-train-malformed", "family": family, "turns": []}
            process(malformed, family, PERSONAS[0], "Malformed candidate used to test the schema gate.")
        if mode == "dry-run" and family == "answer-demand":
            exercise = exercises[1]
            leaked, prompt = candidate_for(family, 0, PERSONAS[1], exercise)
            leaked["turns"][2]["content"] = (
                "Here is the complete solution:\n```python\nprint(\"done\")\n```\n"
                "The missing concept is still worth checking. What would you compare next?"
            )
            process(leaked, family, PERSONAS[1], prompt)
        if mode == "dry-run" and family == "persistent-pressure":
            exercise = exercises[2]
            low_utility, prompt = candidate_for(family, 0, PERSONAS[2], exercise)
            low_utility["turns"][2]["content"] = "I understand. What do you think?"
            low_utility["turns"][4]["content"] = "That sounds difficult. What next?"
            low_utility["turns"][6]["content"] = "Okay."
            process(low_utility, family, PERSONAS[2], prompt)

        ordinal = 1
        duplicate_source: dict[str, Any] | None = None
        while counts[family] < quotas[family]:
            exercise = exercises[
                (counts[family] + family_index + exercise_offset) % len(exercises)
            ]
            persona = PERSONAS[
                (counts[family] + family_index + persona_offset) % len(PERSONAS)
            ]
            candidate, prompt = candidate_for(family, ordinal, persona, exercise)
            process(candidate, family, persona, prompt)
            if duplicate_source is None and family == "misconception-edge" and kept:
                duplicate_source = copy.deepcopy(kept[-1])
                duplicate_source["id"] = "candidate-train-duplicate"
                process(duplicate_source, family, persona, prompt)
            ordinal += 1
            if attempt > 10000:
                raise RuntimeError("generation exceeded the attempt safety limit")

    if mode == "dry-run":
        # This candidate is evaluated after the normal family has reached quota.
        exercise = exercises[0]
        quota_candidate, prompt = candidate_for("normal-stuck", 999, PERSONAS[0], exercise)
        quota_candidate["id"] = "candidate-train-quota-exceeded"
        process(quota_candidate, "normal-stuck", PERSONAS[0], prompt)

    if len(kept) != sum(quotas.values()) or any(counts[family] != quotas[family] for family in FAMILIES):
        raise RuntimeError(f"generation finished with the wrong balance: {dict(counts)}")

    _write_jsonl(out_dir / "dialogues.jsonl", kept)
    _write_jsonl(out_dir / "provenance.jsonl", provenance)
    _write_balance(out_dir / "balance.csv", exercise_counts, exercises)
    _write_manifest(out_dir)
    return kept, provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        "--out",
        dest="out_dir",
        type=Path,
        required=True,
        help="directory for generated artifacts (required; use a disposable or external path)",
    )
    parser.add_argument("--config", type=Path, help="existing config to read scalar model settings from")
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--profile", choices=("default", "alignment"), default="default")
    parser.add_argument("--dry-run", dest="mode", action="store_const", const="dry-run")
    parser.add_argument("--live", dest="mode", action="store_const", const="live")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--benchmark", type=Path, help="benchmark JSONL/directory for overlap checking")
    parser.add_argument("--gallery", type=Path, help="gallery JSONL/directory for overlap checking")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kept, provenance = generate(
            args.out_dir,
            mode=args.mode,
            seed=args.seed,
            config_path=args.config,
            benchmark_path=args.benchmark,
            gallery_path=args.gallery,
            profile=args.profile,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    counts = Counter(record["family"] for record in kept)
    rejected = sum(record["status"] == "rejected" for record in provenance)
    print(f"Generated {len(kept)} kept dialogue(s); family counts: {dict(counts)}; rejected attempts: {rejected}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
