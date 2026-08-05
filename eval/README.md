# Evaluation harness

The harness runs the benchmark cases against a model, records complete chat
transcripts, asks a pinned judge for binary scores, and writes an `EvalResult`
record. It uses only the Python standard library.

## Evaluate

From the repository root:

```bash
python3 -m eval evaluate \
  --model MODEL \
  --cases data/benchmark/cases.jsonl \
  --out runs/base \
  --judge-version judge-v1
```

The command writes:

- `transcripts.jsonl` — every system, learner, and tutor message in order,
  preserving structured assistant message fields, including model id,
  temperature, and seed.
- `judge_calls.jsonl` — every judge request prompt and raw response, plus the
  parsed result, judge model, version, and temperature (`0.0`).
- `eval_results.jsonl` — one record matching `schemas/eval_result.schema.json`.

The evaluator sends the accumulating conversation (the fixed tutor system
prompt plus each learner turn and prior tutor replies) to the model. `--seed`
defaults to `0`; model temperature defaults to `0.0`. The judge always runs at
`0.0`, regardless of the model temperature.

Non-`fixture-*` model ids use the standard-library OpenAI-compatible
`/chat/completions` endpoint. Set `SOCRATIC_API_KEY` (or `OPENAI_API_KEY`) and,
when needed, `SOCRATIC_API_BASE` (or pass `--base-url`). A production run must
pass a separate pinned `--judge-model` or set `SOCRATIC_JUDGE_MODEL`; the
harness refuses to let the evaluated model judge itself. `fixture-clean`,
`fixture-leaking`, and `fixture-judge` make deterministic offline smoke runs.

## Calibration

Export 15–20 transcripts (the default is 15) for human labeling:

```bash
python3 scripts/calibration_export.py \
  --transcripts runs/base/transcripts.jsonl \
  --judge-calls runs/base/judge_calls.jsonl \
  --out data/calibration/labels.jsonl
```

The labels file contains only an opaque calibration id, transcript messages,
and blank human score fields. It does not contain `model_id` or adapter
metadata. Export also writes the private score sidecar
`labels.jsonl.judge.jsonl`.

After the human fills `leakage` and `actionable_diagnosis` with `0` or `1`,
import the labels:

```bash
python3 scripts/calibration_import.py \
  --labels data/calibration/labels.jsonl \
  --judge-calls data/calibration/labels.jsonl.judge.jsonl \
  --out data/calibration/agreement.json
```

The importer compares both binary criteria, writes the agreement report, and
returns a non-zero exit status below 90%. The report includes overall and
per-criterion agreement.

## Local checks

```bash
python3 -m unittest discover -s tests -v
python3 -m eval evaluate --model fixture-clean \
  --cases data/fixtures/benchmark_cases.jsonl \
  --out /tmp/socratic-eval --judge-version fixture-judge-1 \
  --judge-model fixture-judge
python3 scripts/validate_data.py /tmp/socratic-eval/eval_results.jsonl \
  --kind eval_result --skip-manifest
```

The lexical compliance and utility functions in `eval.scoring` are conservative
local smoke checks. The pinned judge is the authority for semantic equivalence
and actionable diagnosis.
