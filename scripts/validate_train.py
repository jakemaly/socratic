#!/usr/bin/env python3
"""Validate the 400-record synthetic training pool and its audit artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .generate_dialogues import check_pool_overlap
    from .gates import REASON_CODES, score_dialogue
    from .training_contract import validate_training_rows
    from .validate_data import BY_KIND, load_jsonl, validate_dataset, validate_relationships
except ImportError:  # pragma: no cover - exercised by the documented CLI.
    from generate_dialogues import check_pool_overlap  # type: ignore[no-redef]
    from gates import REASON_CODES, score_dialogue  # type: ignore[no-redef]
    from training_contract import validate_training_rows  # type: ignore[no-redef]
    from validate_data import BY_KIND, load_jsonl, validate_dataset, validate_relationships  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ARTIFACTS = ("dialogues.jsonl", "exercises.jsonl", "provenance.jsonl")
REQUIRED_ARTIFACTS = (*MANIFEST_ARTIFACTS, "balance.csv", "generator.yaml")
FAMILIES = (
    "normal-stuck",
    "answer-demand",
    "persistent-pressure",
    "misconception-edge",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_manifest(directory: Path) -> list[str]:
    manifest = directory / "manifest.sha256"
    if not manifest.exists():
        return [f"{manifest}: missing"]
    found: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]) is None:
            errors.append(f"{manifest}:{line_number}: expected '<sha256>  <file>'")
            continue
        name = parts[1].lstrip("*").strip()
        path = directory / name
        if name in found:
            errors.append(f"{manifest}:{line_number}: duplicate {name!r}")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append(f"{manifest}:{line_number}: unsafe path {name!r}")
        elif not path.exists():
            errors.append(f"{manifest}:{line_number}: missing artifact {name!r}")
        found[name] = parts[0].lower()

    for name in REQUIRED_ARTIFACTS:
        if not (directory / name).exists():
            errors.append(f"{directory}: missing artifact {name!r}")
    for name in MANIFEST_ARTIFACTS:
        if name not in found:
            errors.append(f"{manifest}: missing entry for {name!r}")
        elif (directory / name).exists() and found[name] != _sha256(directory / name):
            errors.append(f"{manifest}: checksum mismatch for {name!r}")
    return errors


def _check_balance(path: Path, dialogues: list[dict[str, Any]], exercises: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    expected_exercises = [str(exercise["id"]) for exercise in exercises]
    counts: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    for dialogue in dialogues:
        family = dialogue.get("family")
        exercise_id = dialogue.get("exercise_id")
        if family in counts and isinstance(exercise_id, str):
            counts[family][exercise_id] += 1
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        return [f"{path}: cannot read: {exc}"]
    expected_header = ["family", *expected_exercises, "total"]
    if not rows:
        return [f"{path}: balance table is empty"]
    # DictReader preserves the header and makes a malformed/missing column visible.
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != expected_header:
        errors.append(f"{path}: expected header {expected_header!r}, got {header!r}")

    rows_by_family = {row.get("family"): row for row in rows}
    for family in FAMILIES:
        row = rows_by_family.get(family)
        if row is None:
            errors.append(f"{path}: missing family row {family!r}")
            continue
        for exercise_id in expected_exercises:
            try:
                value = int(row.get(exercise_id, ""))
            except ValueError:
                errors.append(f"{path}: {family}/{exercise_id} is not an integer")
                continue
            if value != counts[family][exercise_id]:
                errors.append(
                    f"{path}: {family}/{exercise_id} says {value}, "
                    f"expected {counts[family][exercise_id]}"
                )
        try:
            total = int(row.get("total", ""))
        except ValueError:
            errors.append(f"{path}: {family}/total is not an integer")
        else:
            if total != sum(counts[family].values()):
                errors.append(f"{path}: {family}/total does not match dialogues")
    total_row = rows_by_family.get("total")
    if total_row is None:
        errors.append(f"{path}: missing total row")
    else:
        for exercise_id in expected_exercises:
            actual = sum(counts[family][exercise_id] for family in FAMILIES)
            try:
                reported = int(total_row.get(exercise_id, ""))
            except ValueError:
                errors.append(f"{path}: total/{exercise_id} is not an integer")
            else:
                if reported != actual:
                    errors.append(f"{path}: total/{exercise_id} says {reported}, expected {actual}")
    return errors


def validate(
    directory: Path,
    benchmark: Path | None,
    gallery: Path | None,
    require_canonical_system_prompt: bool = False,
) -> list[str]:
    errors: list[str] = []
    datasets: dict[str, list[dict[str, Any]]] = {}
    for kind in ("dialogue", "exercise"):
        path = directory / BY_KIND[kind].filename
        records, dataset_errors = validate_dataset(BY_KIND[kind], path)
        datasets[kind] = records
        errors.extend(dataset_errors)
    errors.extend(validate_relationships(datasets))

    dialogues = datasets["dialogue"]
    exercises = datasets["exercise"]
    if require_canonical_system_prompt:
        try:
            validate_training_rows(dialogues)
        except ValueError as exc:
            errors.append(f"training contract: {exc}")
    if len(dialogues) != 400:
        errors.append(f"dialogues: expected exactly 400 records, got {len(dialogues)}")
    family_counts = Counter(record.get("family") for record in dialogues)
    for family in FAMILIES:
        if family_counts[family] != 100:
            errors.append(f"dialogues: {family} expected 100 records, got {family_counts[family]}")
    if not exercises:
        errors.append("exercises: pool manifest is empty")
    if any(exercise.get("pool") != "train" for exercise in exercises):
        errors.append("exercises: every exercise must belong to the train pool")
    for dialogue in dialogues:
        if (
            dialogue.get("gate_results", {}).get("compliance_ok") is not True
            or dialogue.get("gate_results", {}).get("utility_ok") is not True
        ):
            errors.append(f"dialogue {dialogue.get('id')!r}: gate metadata is not both true")
        compliance, utility, reasons = score_dialogue(dialogue)
        if not compliance or not utility:
            errors.append(
                f"dialogue {dialogue.get('id')!r}: scorer rejected kept record with {reasons}"
            )

    provenance, provenance_errors = load_jsonl(directory / "provenance.jsonl")
    errors.extend(provenance_errors)
    provenance_records = [record for _, record in provenance]
    kept_ids = {record.get("id") for record in dialogues}
    kept_audits = {
        record.get("candidate_id")
        for record in provenance_records
        if record.get("status") == "kept"
    }
    if kept_audits != kept_ids:
        errors.append("provenance: kept entries do not exactly cover dialogue ids")
    for record in provenance_records:
        status = record.get("status")
        reasons = record.get("reason_codes")
        if status == "rejected":
            if not isinstance(reasons, list) or not reasons:
                errors.append(f"provenance {record.get('candidate_id')!r}: rejected entry has no reason")
            elif any(reason not in REASON_CODES for reason in reasons):
                errors.append(f"provenance {record.get('candidate_id')!r}: unknown reason code")
        elif status == "kept" and reasons != []:
            errors.append(f"provenance {record.get('candidate_id')!r}: kept entry has rejection reasons")
        elif status not in {"kept", "rejected"}:
            errors.append(f"provenance {record.get('candidate_id')!r}: unknown status")
    rejected_reasons = {
        reason
        for record in provenance_records
        if record.get("status") == "rejected"
        for reason in record.get("reason_codes", [])
    }
    if set(REASON_CODES) - rejected_reasons:
        errors.append(
            "provenance: missing rejection reason examples "
            + ", ".join(sorted(set(REASON_CODES) - rejected_reasons))
        )

    errors.extend(_check_balance(directory / "balance.csv", dialogues, exercises))
    errors.extend(_check_manifest(directory))
    overlaps = check_pool_overlap(exercises, benchmark, gallery)
    errors.extend(f"pool overlap: {message}" for message in overlaps)
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="training-pool directory to validate (required; raw pools are not shipped)",
    )
    parser.add_argument("--benchmark", type=Path, help="benchmark JSONL/directory to check for overlap")
    parser.add_argument("--gallery", type=Path, help="gallery JSONL/directory to check for overlap")
    parser.add_argument(
        "--require-canonical-system-prompt",
        action="store_true",
        help="require the exact evaluation system prompt in every training dialogue",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        errors = validate(
            args.directory,
            args.benchmark,
            args.gallery,
            args.require_canonical_system_prompt,
        )
    except OSError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Validated training pool: 400 dialogues, four 100-record families, pool separation, gates, provenance, and manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
