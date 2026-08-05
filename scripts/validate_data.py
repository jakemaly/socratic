#!/usr/bin/env python3
"""Validate Socratic JSONL datasets and their SHA-256 manifest.

This intentionally uses only the Python standard library so every track can run
it before adding its own dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "data" / "fixtures"
SCHEMA_DIR = ROOT / "schemas"
SCHEMA_VERSION = "1.0.0"
FAMILIES = (
    "normal-stuck",
    "answer-demand",
    "persistent-pressure",
    "misconception-edge",
)
POOLS = ("benchmark", "train", "gallery")


@dataclass(frozen=True)
class DatasetSpec:
    kind: str
    filename: str
    schema_filename: str


DATASETS = (
    DatasetSpec("benchmark_case", "benchmark_cases.jsonl", "benchmark_case.schema.json"),
    DatasetSpec("dialogue", "dialogues.jsonl", "dialogue.schema.json"),
    DatasetSpec("exercise", "exercises.jsonl", "exercise.schema.json"),
    DatasetSpec("eval_result", "eval_results.jsonl", "eval_result.schema.json"),
)
BY_KIND = {spec.kind: spec for spec in DATASETS}
BY_FILENAME = {spec.filename: spec for spec in DATASETS}


# The repository intentionally does not depend on jsonschema. This is the small
# JSON Schema subset used by the four checked-in schemas.
def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"only local schema references are supported: {reference}")
    current: Any = root
    for part in reference[2:].split("/"):
        current = current[part.replace("~1", "/").replace("~0", "~")]
    if not isinstance(current, dict):
        raise ValueError(f"schema reference does not point to an object: {reference}")
    return current


def _display(value: Any) -> str:
    return repr(value) if not isinstance(value, str) else repr(value)


def validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
    root: dict[str, Any] | None = None,
) -> list[str]:
    """Return human-readable errors for the supported JSON Schema keywords."""
    root = schema if root is None else root
    errors: list[str] = []

    if "$ref" in schema:
        try:
            referenced = _resolve_ref(root, schema["$ref"])
        except (KeyError, TypeError, ValueError) as exc:
            return [f"{path}: invalid schema reference: {exc}"]
        return validate_schema_value(value, referenced, path, root)

    if "type" in schema:
        expected_types = schema["type"]
        if isinstance(expected_types, str):
            expected_types = [expected_types]
        if not any(_json_type_matches(value, expected) for expected in expected_types):
            return [f"{path}: expected type {expected_types}, got {type(value).__name__}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {_display(schema['const'])}, got {_display(value)}")

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']}, got {_display(value)}")

    if "required" in schema and isinstance(value, dict):
        for name in schema["required"]:
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for name, child in value.items():
            child_path = f"{path}.{name}"
            if name in properties:
                errors.extend(validate_schema_value(value[name], properties[name], child_path, root))
            elif additional is False:
                errors.append(f"{child_path}: unexpected property")
            elif isinstance(additional, dict):
                errors.extend(validate_schema_value(value[name], additional, child_path, root))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            errors.extend(
                validate_schema_value(item, schema["items"], f"{path}[{index}]", root)
            )

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: must contain at least {schema['minLength']} character(s)")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: must contain at most {schema['maxLength']} character(s)")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: does not match {schema['pattern']!r}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: must contain at least {schema['minItems']} item(s)")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: must contain at most {schema['maxItems']} item(s)")

    if isinstance(value, dict) and "minProperties" in schema:
        if len(value) < schema["minProperties"]:
            errors.append(f"{path}: must contain at least {schema['minProperties']} properties")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be at most {schema['maximum']}")

    if "anyOf" in schema:
        alternatives = [
            validate_schema_value(value, candidate, path, root)
            for candidate in schema["anyOf"]
        ]
        if all(alternative for alternative in alternatives):
            errors.append(f"{path}: does not match any allowed schema")

    if "oneOf" in schema:
        matches = sum(
            not validate_schema_value(value, candidate, path, root)
            for candidate in schema["oneOf"]
        )
        if matches != 1:
            errors.append(f"{path}: must match exactly one allowed schema")

    return errors


def load_json(path: Path) -> tuple[Any | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except OSError as exc:
        return None, [f"{path}: cannot read file: {exc}"]
    except json.JSONDecodeError as exc:
        return None, [f"{path}:{exc.lineno}: invalid JSON: {exc.msg}"]


def load_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    records: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [], [f"{path}: cannot read file: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}:{line_number}: each JSONL record must be an object")
            continue
        records.append((line_number, value))
    return records, errors


def load_schema(spec: DatasetSpec) -> tuple[dict[str, Any] | None, list[str]]:
    path = SCHEMA_DIR / spec.schema_filename
    value, errors = load_json(path)
    if errors:
        return None, errors
    if not isinstance(value, dict):
        return None, [f"{path}: schema root must be an object"]
    return value, []


def validate_dataset(
    spec: DatasetSpec,
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    schema, errors = load_schema(spec)
    records, parse_errors = load_jsonl(path)
    errors.extend(parse_errors)
    if schema is None:
        return [record for _, record in records], errors

    seen_ids: set[str] = set()
    for line_number, record in records:
        record_path = f"{path}:{line_number}"
        for message in validate_schema_value(record, schema):
            errors.append(f"{record_path}: {message}")
        record_id = record.get("id")
        if isinstance(record_id, str):
            if record_id in seen_ids:
                errors.append(f"{record_path}: duplicate id {record_id!r}")
            seen_ids.add(record_id)
    return [record for _, record in records], errors


def _rate(total: int, count: int) -> float:
    return count / total if total else 0.0


def _close(left: Any, right: float) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and abs(left - right) <= 1e-6
    )


def validate_relationships(datasets: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Validate constraints that JSON Schema cannot express across JSONL files."""
    errors: list[str] = []
    exercises = {
        record.get("id"): record
        for record in datasets.get("exercise", [])
        if isinstance(record.get("id"), str)
    }
    cases = {
        record.get("id"): record
        for record in datasets.get("benchmark_case", [])
        if isinstance(record.get("id"), str)
    }

    for dialogue in datasets.get("dialogue", []):
        # A single-file schema check cannot resolve cross-file references.  The
        # relationship check remains strict whenever the exercise dataset was
        # selected (the normal directory/fixture path).
        if "exercise" in datasets:
            exercise_id = dialogue.get("exercise_id")
            exercise = exercises.get(exercise_id)
            if exercise is None:
                errors.append(
                    f"dialogue {dialogue.get('id')!r}: exercise_id {exercise_id!r} does not exist"
                )
            elif dialogue.get("pool") != exercise.get("pool"):
                errors.append(
                    f"dialogue {dialogue.get('id')!r}: pool {dialogue.get('pool')!r} "
                    f"does not match exercise {exercise_id!r} pool {exercise.get('pool')!r}"
                )
        turns = dialogue.get("turns")
        roles = {
            turn.get("role")
            for turn in turns
            if isinstance(turn, dict)
        } if isinstance(turns, list) else set()
        if "user" not in roles or "assistant" not in roles:
            errors.append(
                f"dialogue {dialogue.get('id')!r}: turns must include both user and assistant roles"
            )

    for result in datasets.get("eval_result", []):
        raw_scores = result.get("case_scores", [])
        scores = raw_scores if isinstance(raw_scores, list) else []
        seen_case_ids: set[str] = set()
        family_counts: Counter[str] = Counter()
        leakage_total = 0
        diagnosis_total = 0
        for score in scores:
            if not isinstance(score, dict):
                continue
            case_id = score.get("case_id")
            if isinstance(case_id, str):
                if case_id in seen_case_ids:
                    errors.append(
                        f"eval result {result.get('run_id')!r}: duplicate case_id {case_id!r}"
                    )
                seen_case_ids.add(case_id)
            case = cases.get(case_id)
            if case is not None and score.get("family") != case.get("family"):
                errors.append(
                    f"eval result {result.get('run_id')!r}: family for {case_id!r} "
                    f"does not match benchmark case"
                )
            family = score.get("family")
            if family in FAMILIES:
                family_counts[family] += 1
            leakage = score.get("leakage")
            if isinstance(leakage, int) and not isinstance(leakage, bool):
                leakage_total += leakage
            diagnosis = score.get("actionable_diagnosis")
            if isinstance(diagnosis, int) and not isinstance(diagnosis, bool):
                diagnosis_total += diagnosis

        raw_aggregates = result.get("aggregates", {})
        aggregates = raw_aggregates if isinstance(raw_aggregates, dict) else {}
        if aggregates.get("case_count") != len(scores):
            errors.append(
                f"eval result {result.get('run_id')!r}: aggregates.case_count does not match case_scores"
            )
        if scores and isinstance(aggregates.get("leakage_rate"), (int, float)):
            if not _close(aggregates["leakage_rate"], _rate(len(scores), leakage_total)):
                errors.append(
                    f"eval result {result.get('run_id')!r}: leakage_rate does not match case scores"
                )
        if scores and isinstance(aggregates.get("diagnosis_rate"), (int, float)):
            if not _close(aggregates["diagnosis_rate"], _rate(len(scores), diagnosis_total)):
                errors.append(
                    f"eval result {result.get('run_id')!r}: diagnosis_rate does not match case scores"
                )

        raw_by_family = aggregates.get("by_family", {})
        by_family = raw_by_family if isinstance(raw_by_family, dict) else {}
        for family in FAMILIES:
            raw_family_result = by_family.get(family, {})
            family_result = (
                raw_family_result if isinstance(raw_family_result, dict) else {}
            )
            expected_count = family_counts[family]
            if family_result.get("count") != expected_count:
                errors.append(
                    f"eval result {result.get('run_id')!r}: by_family[{family!r}].count "
                    "does not match case scores"
                )
            family_scores = [
                score for score in scores
                if isinstance(score, dict) and score.get("family") == family
            ]
            valid_family_scores = [
                score for score in family_scores
                if isinstance(score.get("leakage"), int)
                and not isinstance(score.get("leakage"), bool)
                and isinstance(score.get("actionable_diagnosis"), int)
                and not isinstance(score.get("actionable_diagnosis"), bool)
            ]
            if valid_family_scores:
                family_leakage = sum(score["leakage"] for score in valid_family_scores)
                family_diagnosis = sum(
                    score["actionable_diagnosis"] for score in valid_family_scores
                )
                if not _close(
                    family_result.get("leakage_rate", -1),
                    _rate(len(valid_family_scores), family_leakage),
                ):
                    errors.append(
                        f"eval result {result.get('run_id')!r}: by_family[{family!r}].leakage_rate "
                        "does not match case scores"
                    )
                if not _close(
                    family_result.get("diagnosis_rate", -1),
                    _rate(len(valid_family_scores), family_diagnosis),
                ):
                    errors.append(
                        f"eval result {result.get('run_id')!r}: by_family[{family!r}].diagnosis_rate "
                        "does not match case scores"
                    )

    return errors


def validate_benchmark_balance(
    cases: list[dict[str, Any]],
    require_balanced: bool,
) -> list[str]:
    if not require_balanced:
        return []
    counts = Counter(
        case.get("family")
        for case in cases
        if isinstance(case.get("family"), str)
    )
    expected = {family: counts[family] for family in FAMILIES}
    if len(cases) != 48 or any(expected[family] != 12 for family in FAMILIES):
        return [
            "benchmark balance: expected 48 cases with 12 cases in each family; "
            f"got {len(cases)} cases with counts {dict(expected)}"
        ]
    return []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(directory: Path, manifest_path: Path) -> None:
    files = sorted(directory.glob("*.jsonl"))
    manifest_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def validate_manifest(directory: Path, manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return [f"{manifest_path}: SHA-256 manifest is missing"]
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"{manifest_path}: cannot read manifest: {exc}"]

    expected_files = {path.name: path for path in directory.glob("*.jsonl")}
    found: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            errors.append(f"{manifest_path}:{line_number}: expected '<sha256>  <file>'")
            continue
        filename = parts[1].lstrip("*").strip()
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"{manifest_path}:{line_number}: unsafe file name {filename!r}")
            continue
        if filename in found:
            errors.append(f"{manifest_path}:{line_number}: duplicate file {filename!r}")
        found[filename] = parts[0].lower()

    for filename in sorted(set(found) - set(expected_files)):
        errors.append(f"{manifest_path}: entry names missing or unknown file {filename!r}")
    for filename in sorted(set(expected_files) - set(found)):
        errors.append(f"{manifest_path}: missing entry for {filename!r}")
    for filename, path in expected_files.items():
        actual = sha256_file(path)
        if found.get(filename) != actual:
            errors.append(f"{manifest_path}: checksum mismatch for {filename!r}")
    return errors


def select_datasets(path: Path, kind: str | None) -> tuple[dict[str, tuple[DatasetSpec, Path]], list[str]]:
    errors: list[str] = []
    selected: dict[str, tuple[DatasetSpec, Path]] = {}
    if path.is_file():
        if kind is None:
            spec = BY_FILENAME.get(path.name)
            if spec is None:
                errors.append(
                    f"{path}: cannot infer dataset kind; pass --kind with one of "
                    f"{', '.join(BY_KIND)}"
                )
                return selected, errors
        else:
            spec = BY_KIND[kind]
        selected[spec.kind] = (spec, path)
        return selected, errors

    if not path.is_dir():
        return selected, [f"{path}: directory or JSONL file does not exist"]
    if kind is not None:
        file_path = path / BY_KIND[kind].filename
        if not file_path.exists():
            return selected, [f"{file_path}: expected JSONL file is missing"]
        selected[kind] = (BY_KIND[kind], file_path)
        return selected, errors

    for spec in DATASETS:
        file_path = path / spec.filename
        if not file_path.exists():
            errors.append(f"{file_path}: expected JSONL file is missing")
        else:
            selected[spec.kind] = (spec, file_path)
    return selected, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_FIXTURES,
        help="fixture directory or one JSONL file (default: data/fixtures)",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(BY_KIND),
        help="schema kind when validating a file with a non-canonical name",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: SHA256SUMS beside the JSONL files)",
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="skip checksum validation (useful for local negative tests)",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="rewrite the manifest before validating it",
    )
    parser.add_argument(
        "--require-balanced-benchmark",
        action="store_true",
        help="require the full 48-case, four-by-twelve benchmark balance",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    selected, selection_errors = select_datasets(args.path, args.kind)
    errors.extend(selection_errors)

    datasets: dict[str, list[dict[str, Any]]] = {}
    for kind, (spec, path) in selected.items():
        records, dataset_errors = validate_dataset(spec, path)
        datasets[kind] = records
        errors.extend(dataset_errors)

    if selected:
        errors.extend(validate_relationships(datasets))
        errors.extend(
            validate_benchmark_balance(
                datasets.get("benchmark_case", []), args.require_balanced_benchmark
            )
        )

    directory = args.path if args.path.is_dir() else args.path.parent
    manifest_path = args.manifest or directory / "SHA256SUMS"
    if args.write_manifest:
        try:
            write_manifest(directory, manifest_path)
        except OSError as exc:
            errors.append(f"{manifest_path}: cannot write manifest: {exc}")
    if not args.skip_manifest:
        errors.extend(validate_manifest(directory, manifest_path))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    record_count = sum(len(records) for records in datasets.values())
    kinds = ", ".join(
        f"{len(datasets[kind])} {kind.replace('_', ' ')} record(s)"
        for kind in (spec.kind for spec in DATASETS)
        if kind in datasets
    )
    manifest_message = " (manifest checked)" if not args.skip_manifest else ""
    print(f"Validated {record_count} record(s): {kinds}{manifest_message}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
