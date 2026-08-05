"""Command-line entry points for evaluation and calibration."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from .calibration import (
    CalibrationGateError,
    export_calibration,
    import_calibration,
)
from .client import make_judge_model, make_model
from .io import read_jsonl, write_jsonl
from .judge import JudgeClient
from .runner import evaluate_cases


def _evaluate(args: argparse.Namespace) -> int:
    cases = read_jsonl(Path(args.cases))
    out = Path(args.out)
    model = make_model(args.model, base_url=args.base_url)
    judge_model_id = args.judge_model or os.getenv("SOCRATIC_JUDGE_MODEL")
    if not judge_model_id:
        if args.model.lower().startswith("fixture"):
            judge_model_id = "fixture-judge"
        else:
            raise ValueError(
                "a separate pinned judge is required; pass --judge-model "
                "or set SOCRATIC_JUDGE_MODEL"
            )
    if judge_model_id == args.model:
        raise ValueError("judge model must be separate from the model under evaluation")
    judge = JudgeClient(
        make_judge_model(judge_model_id, base_url=args.base_url),
        model_id=judge_model_id,
        version=args.judge_version,
    )
    try:
        evaluation = evaluate_cases(
            cases,
            model,
            model_id=args.model,
            judge=judge,
            run_id=args.run_id,
            adapter_id=args.adapter_id,
            temperature=args.temperature,
            seed=args.seed,
        )
    except Exception:
        # A failed judge response is still an auditable invocation.
        if judge.calls:
            write_jsonl(out / "judge_calls.jsonl", judge.calls)
        raise

    write_jsonl(out / "transcripts.jsonl", [transcript.to_dict() for transcript in evaluation.transcripts])
    write_jsonl(out / "judge_calls.jsonl", evaluation.judge_calls)
    write_jsonl(out / "eval_results.jsonl", [evaluation.report])
    aggregates = evaluation.report["aggregates"]
    print(
        f"Evaluated {aggregates['case_count']} case(s): "
        f"leakage={aggregates['leakage_rate']:.1%}, "
        f"utility={aggregates['diagnosis_rate']:.1%}."
    )
    return 0


def _calibration_export(args: argparse.Namespace) -> int:
    labels = export_calibration(
        Path(args.transcripts),
        Path(args.out),
        Path(args.judge_scores) if args.judge_scores else None,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Exported {len(labels)} blinded calibration transcript(s) to {args.out}.")
    return 0


def _calibration_import(args: argparse.Namespace) -> int:
    try:
        report = import_calibration(
            Path(args.labels),
            Path(args.judge_scores) if args.judge_scores else None,
            Path(args.out) if args.out else None,
            threshold=args.threshold,
        )
    except CalibrationGateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Calibration agreement: {report['agreement_rate']:.1%} (passed).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser("evaluate", help="run benchmark cases")
    evaluate.add_argument("--model", required=True, help="model id under test")
    evaluate.add_argument("--cases", required=True, help="benchmark cases JSONL")
    evaluate.add_argument("--out", required=True, help="output directory")
    evaluate.add_argument("--judge-version", required=True, help="pinned judge rubric/model version")
    evaluate.add_argument("--judge-model", help="separate pinned strong judge model id (or set SOCRATIC_JUDGE_MODEL)")
    evaluate.add_argument("--adapter-id", help="adapter or fine-tuning id")
    evaluate.add_argument("--base-url", help="OpenAI-compatible API base URL")
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--run-id")
    evaluate.set_defaults(handler=_evaluate)

    calibration = commands.add_parser("calibration", help="export or import blinded judge calibration")
    calibration_commands = calibration.add_subparsers(dest="calibration_command", required=True)

    export = calibration_commands.add_parser("export", help="export a blinded labels template")
    export.add_argument("--transcripts", required=True)
    export.add_argument("--judge-scores", "--judge-calls", dest="judge_scores")
    export.add_argument("--out", required=True, help="labels JSONL output")
    export.add_argument("--sample-size", type=int, default=15)
    export.add_argument("--seed", type=int, default=0)
    export.set_defaults(handler=_calibration_export)

    import_command = calibration_commands.add_parser("import", help="check returned labels against judge scores")
    import_command.add_argument("--labels", required=True)
    import_command.add_argument("--judge-scores", "--judge-calls", dest="judge_scores")
    import_command.add_argument("--out", help="agreement JSON output")
    import_command.add_argument("--threshold", type=float, default=0.9)
    import_command.set_defaults(handler=_calibration_import)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
