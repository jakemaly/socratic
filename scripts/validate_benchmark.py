#!/usr/bin/env python3
"""Validate the curated benchmark cases and their exercise-pool separation."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data" / "benchmark"
DEFAULT_CASES = BENCHMARK_DIR / "cases.jsonl"

GALLERY_EXERCISES = (
    "Greeting",
    "Even-or-odd",
    "Countdown",
    "Average",
    "Temp converter",
    "Word counter",
)
FAMILIES = (
    "normal-stuck",
    "answer-demand",
    "persistent-pressure",
    "misconception-edge",
)
EXERCISE_RE = re.compile(r"^Exercise:\s*([^.;]+?)\.\s")

# Reuse the repository's stdlib-only schema implementation rather than making a
# second benchmark validator with subtly different rules.
sys.path.insert(0, str(ROOT / "scripts"))
from validate_data import (  # noqa: E402
    BY_KIND,
    validate_benchmark_balance,
    validate_dataset,
    validate_manifest,
)


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def exercise_name(scenario: str) -> str | None:
    match = EXERCISE_RE.match(scenario)
    return match.group(1).strip() if match else None


def load_names(path: Path) -> list[str]:
    """Read exercise names from a small JSONL or line-oriented manifest.

    JSONL manifests may use name/title/exercise/exercise_name, or a prompt.
    Plain text manifests use one exercise per line; markdown bullets and table
    separators are ignored. This keeps the check usable while the training
    pipeline settles on its final manifest shape.
    """
    if not path.exists():
        return []

    names: list[str] = []
    if path.suffix.lower() in {".json", ".jsonl"}:
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".jsonl":
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                raw = json.loads(text)
                values = raw if isinstance(raw, list) else [raw]
        except (OSError, json.JSONDecodeError):
            values = []
        for value in values:
            if isinstance(value, str):
                names.append(value)
            elif isinstance(value, dict):
                for key in ("name", "title", "exercise", "exercise_name", "prompt"):
                    candidate = value.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        names.append(candidate.strip())
                        break
        if values:
            return names

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        value = line.strip().strip("-*").strip()
        if not value or value.startswith("#") or set(value) <= {"|", ":", "-", " "}:
            continue
        if value.startswith("|"):
            cells = [cell.strip() for cell in value.strip("|").split("|")]
            value = cells[0] if cells else ""
        if value and value.lower() not in {"exercise", "exercise name", "name"}:
            names.append(value.rstrip("."))
    return names


def overlaps(left: str, right: str) -> bool:
    a, b = normalise(left), normalise(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def default_training_manifests() -> list[Path]:
    return [
        ROOT / "data" / "train" / "exercises.jsonl",
        ROOT / "data" / "train" / "exercise_pool.jsonl",
        ROOT / "data" / "train" / "exercises.txt",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="?",
        type=Path,
        default=DEFAULT_CASES,
        help="benchmark JSONL path (default: data/benchmark/cases.jsonl)",
    )
    parser.add_argument(
        "--train-exercises",
        action="append",
        type=Path,
        help="training exercise manifest; repeat for multiple manifests",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="SHA-256 manifest (default: SHA256SUMS beside cases)",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="skip checksum validation for a temporary or copied case file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases_path = args.cases.resolve()
    errors: list[str] = []

    records, schema_errors = validate_dataset(BY_KIND["benchmark_case"], cases_path)
    errors.extend(schema_errors)
    errors.extend(validate_benchmark_balance(records, require_balanced=True))

    counts = Counter(record.get("family") for record in records)
    print(f"Cases: {len(records)}")
    for family in FAMILIES:
        print(f"  {family}: {counts[family]}")

    exercises: list[str] = []
    for index, record in enumerate(records, start=1):
        case_id = record.get("id")
        family = record.get("family")
        if isinstance(case_id, str) and isinstance(family, str):
            family_number = sum(
                1 for earlier in records[:index] if earlier.get("family") == family
            )
            expected_id = f"case-{family}-{family_number:03d}"
            if case_id != expected_id:
                errors.append(
                    f"{cases_path}: case {index} has id {case_id!r}; "
                    f"expected deterministic id {expected_id!r}"
                )
        turns = record.get("learner_turns")
        if isinstance(turns, list) and not 3 <= len(turns) <= 6:
            errors.append(
                f"{cases_path}: {case_id!r} must have 3-6 learner turns, got {len(turns)}"
            )
        scenario = record.get("scenario")
        name = exercise_name(scenario) if isinstance(scenario, str) else None
        if name is None:
            errors.append(
                f"{cases_path}: {case_id!r} scenario must start with 'Exercise: <name>.'"
            )
        else:
            exercises.append(name)

    exercise_counts = Counter(exercises)
    print(f"Exercises: {len(exercise_counts)} ({sum(exercise_counts.values())} cases)")
    for name in sorted(exercise_counts):
        print(f"  {name}: {exercise_counts[name]}")

    collisions: list[str] = []
    for benchmark_name in exercise_counts:
        for gallery_name in GALLERY_EXERCISES:
            if overlaps(benchmark_name, gallery_name):
                collisions.append(f"gallery: {benchmark_name} overlaps {gallery_name}")

    train_paths = [path.resolve() for path in args.train_exercises] if args.train_exercises else [
        path for path in default_training_manifests() if path.exists()
    ]
    train_names: list[str] = []
    for path in train_paths:
        train_names.extend(load_names(path))
    for benchmark_name in exercise_counts:
        for train_name in train_names:
            if overlaps(benchmark_name, train_name):
                collisions.append(f"train: {benchmark_name} overlaps {train_name}")

    print("Pool collisions:")
    gallery_collisions = [item for item in collisions if item.startswith("gallery:")]
    train_collisions = [item for item in collisions if item.startswith("train:")]
    print(f"  gallery: {len(gallery_collisions)}")
    print(f"  train: {len(train_collisions)}")
    for collision in collisions:
        print(f"    - {collision}")
    if not train_paths:
        print("  train manifest: not present; pass --train-exercises when #9 publishes it")
    elif not train_names:
        print("  train manifest: present but no exercise names were read")
    else:
        print(f"  train manifest: {len(train_names)} exercise name(s) checked")
    errors.extend(f"pool collision: {collision}" for collision in collisions)

    if not args.skip_manifest:
        manifest_path = (args.manifest or cases_path.parent / "SHA256SUMS").resolve()
        errors.extend(validate_manifest(cases_path.parent, manifest_path))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Benchmark self-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
