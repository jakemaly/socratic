"""Blinded calibration export/import and the 90% agreement gate."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json, write_jsonl


class CalibrationGateError(RuntimeError):
    """Raised when imported labels do not meet the calibration threshold."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(f"calibration agreement {report['agreement_rate']:.1%} is below 90%")


def _binary(record: dict[str, Any], name: str) -> int | None:
    candidates = [name, f"human_{name}"]
    for container_name in ("labels", "human_labels"):
        container = record.get(container_name)
        if isinstance(container, dict):
            candidates.append((container_name, name))
    for candidate in candidates:
        if isinstance(candidate, tuple):
            value = record[candidate[0]].get(candidate[1])
        else:
            value = record.get(candidate)
        if value is not None:
            if isinstance(value, bool) or value not in (0, 1):
                raise ValueError(f"{name} must be integer 0 or 1")
            return int(value)
    return None


def _record_key(record: dict[str, Any]) -> str | None:
    for key in ("calibration_id", "id", "transcript_id", "calibration_source"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _judge_result(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("result", "judge", "scores"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return record


def _sidecar(path: Path) -> Path:
    return Path(f"{path}.judge.jsonl")


def export_calibration(
    transcripts_path: Path,
    out_path: Path,
    judge_scores_path: Path | None = None,
    *,
    sample_size: int = 15,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Write a model-blinded labels template and a private judge-score sidecar."""
    if sample_size < 1 or sample_size > 20:
        raise ValueError("calibration sample size must be between 1 and 20")
    transcripts = read_jsonl(transcripts_path)
    if sample_size > len(transcripts):
        raise ValueError(
            f"requested {sample_size} calibration transcripts, but only {len(transcripts)} exist"
        )
    judge_records = read_jsonl(judge_scores_path) if judge_scores_path else []
    judge_by_key = {
        key: _judge_result(record)
        for record in judge_records
        if (key := _record_key(record)) is not None
    }

    selected_indexes = sorted(random.Random(seed).sample(range(len(transcripts)), sample_size))
    labels: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    for number, index in enumerate(selected_indexes, start=1):
        transcript = transcripts[index]
        transcript_id = transcript.get("transcript_id")
        messages = transcript.get("messages")
        if not isinstance(transcript_id, str) or not isinstance(messages, list):
            raise ValueError("each transcript needs transcript_id and messages")
        calibration_id = f"cal-{number:03d}"
        labels.append(
            {
                "schema_version": "1.0.0",
                "calibration_id": calibration_id,
                "transcript_id": transcript_id,
                "messages": messages,
                "leakage": None,
                "actionable_diagnosis": None,
            }
        )
        judge = judge_by_key.get(transcript_id, {})
        sidecar.append(
            {
                "schema_version": "1.0.0",
                "calibration_id": calibration_id,
                "transcript_id": transcript_id,
                "leakage": judge.get("leakage"),
                "actionable_diagnosis": judge.get("actionable_diagnosis"),
            }
        )

    write_jsonl(out_path, labels)
    write_jsonl(_sidecar(out_path), sidecar)
    return labels


def import_calibration(
    labels_path: Path,
    judge_scores_path: Path | None = None,
    out_path: Path | None = None,
    *,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Compare returned human labels with judge labels and write the gate report."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    labels = read_jsonl(labels_path)
    scores_path = judge_scores_path or _sidecar(labels_path)
    if not scores_path.exists():
        raise ValueError(f"judge score file does not exist: {scores_path}")
    judge_records = read_jsonl(scores_path)
    judge_by_key = {
        key: _judge_result(record)
        for record in judge_records
        if (key := _record_key(record)) is not None
    }
    if len(judge_by_key) != len(judge_records):
        raise ValueError("every judge score record needs a calibration or transcript id")

    matches = 0
    comparisons = 0
    by_metric: dict[str, dict[str, int | float]] = {}
    for metric in ("leakage", "actionable_diagnosis"):
        metric_matches = 0
        metric_total = 0
        for label in labels:
            key = _record_key(label)
            if key is None:
                raise ValueError("every calibration label needs an id")
            judge = judge_by_key.get(key)
            if judge is None and isinstance(label.get("transcript_id"), str):
                judge = judge_by_key.get(label["transcript_id"])
            if judge is None:
                raise ValueError(f"missing judge score for {key!r}")
            human_value = _binary(label, metric)
            judge_value = _binary(judge, metric)
            if human_value is None:
                raise ValueError(f"missing human {metric} label for {key!r}")
            if judge_value is None:
                raise ValueError(f"missing judge {metric} label for {key!r}")
            metric_total += 1
            comparisons += 1
            if human_value == judge_value:
                metric_matches += 1
                matches += 1
        by_metric[metric] = {
            "matches": metric_matches,
            "total": metric_total,
            "agreement_rate": round(metric_matches / metric_total, 6)
            if metric_total
            else 0.0,
        }

    agreement_rate = matches / comparisons if comparisons else 0.0
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "sample_count": len(labels),
        "label_count": comparisons,
        "matches": matches,
        "agreement_rate": round(agreement_rate, 6),
        "threshold": threshold,
        "by_metric": by_metric,
        "passed": agreement_rate >= threshold,
    }
    if out_path:
        write_json(out_path, report)
    if not report["passed"]:
        raise CalibrationGateError(report)
    return report
