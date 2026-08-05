# Dataset schemas

This directory defines the versioned data contract shared by the benchmark, synthetic-dialogue, gallery, and evaluation tracks.

## Records

| Schema | JSONL record | Purpose |
| --- | --- | --- |
| `benchmark_case.schema.json` | `benchmark_cases.jsonl` | A multi-turn benchmark scenario and the behavior the tutor should exhibit. |
| `dialogue.schema.json` | `dialogues.jsonl` | A gated conversation from one data pool. |
| `exercise.schema.json` | `exercises.jsonl` | An exercise assigned to one pool. |
| `eval_result.schema.json` | `eval_results.jsonl` | Run metadata, per-case binary scores, and aggregate rates. |

Every record has `schema_version: "1.0.0"`. JSONL means one JSON object per line; blank lines are ignored. The schemas deliberately keep benchmark, training, and gallery content distinguishable with the shared `pool` enum (`benchmark`, `train`, `gallery`). A dialogue's `exercise_id` must refer to an exercise in the same pool.

The four benchmark families are:

- `normal-stuck`
- `answer-demand`
- `persistent-pressure`
- `misconception-edge`

`EvalResult.case_scores` stores binary `leakage` and `actionable_diagnosis` values as `0` or `1`. Its `aggregates.by_family` object records counts and rates for all four families, including families not present in a small fixture.

## Versioning and integrity

1. Add or change JSONL records without reformatting existing lines unnecessarily.
2. Keep records at `schema_version: "1.0.0"` for compatible data-only changes. Bump the schema version and update all four schemas for a contract change.
3. Regenerate `data/fixtures/SHA256SUMS` whenever a JSONL file changes. The manifest covers the exact JSONL bytes and is checked by the validator.
4. Validate before handing data to another track:

```bash
python scripts/validate_data.py
```

The validator uses only Python's standard library. It checks each record against its schema, required cross-file references, pool separation, aggregate consistency, duplicate IDs, and the SHA-256 manifest. A full benchmark run can additionally enforce the planned four-by-twelve balance:

```bash
python scripts/validate_data.py data/benchmark --require-balanced-benchmark
```

The synthetic training pool is validated as a combined dialogue/exercise set so
its cross-file references, 100-per-family quotas, provenance, overlap checks,
and balance table are checked together:

```bash
python3 scripts/validate_train.py data/train
```

The committed fixtures are intentionally small (3 cases, 3 dialogues, 1 exercise, and 1 evaluation result), so the balance flag is not used for them.
