# Dataset schemas

This directory defines the versioned data contract shared by the benchmark, synthetic-dialogue, and evaluation tracks.

## Records

| Schema | JSONL record | Purpose |
| --- | --- | --- |
| `benchmark_case.schema.json` | `benchmark_cases.jsonl` | A multi-turn benchmark scenario and the behavior the tutor should exhibit. |
| `dialogue.schema.json` | `dialogues.jsonl` | A gated conversation from one data pool. |
| `exercise.schema.json` | `exercises.jsonl` | An exercise assigned to one pool. |
| `eval_result.schema.json` | `eval_results.jsonl` | Run metadata, per-case binary scores, and aggregate rates. |

Every dataset record has `schema_version: "1.0.0"`. JSONL means one JSON object per line; blank lines are ignored. The dataset schemas deliberately keep benchmark and training content distinguishable with the shared `pool` enum (`benchmark`, `train`, `gallery`). A dialogue's `exercise_id` must refer to an exercise in the same pool.

The four benchmark families are:

- `normal-stuck`
- `answer-demand`
- `persistent-pressure`
- `misconception-edge`

`EvalResult.case_scores` stores binary `leakage` and `actionable_diagnosis` values as `0` or `1`. Its `aggregates.by_family` object records counts and rates for all four families, including families not present in a small fixture.

## Versioning and integrity

1. Add or change JSONL records without reformatting existing lines unnecessarily.
2. Keep dataset records at `schema_version: "1.0.0"` for compatible data-only changes. Bump the schema version and update all four dataset schemas for a contract change.
3. Regenerate `data/fixtures/SHA256SUMS` whenever a JSONL file changes. The manifest covers the exact JSONL bytes and is checked by the validator.
4. Validate before handing data to another track:

```bash
python3 scripts/validate_data.py
```

The validator uses only Python's standard library. It checks each fixture record against its schema, required cross-file references, pool separation, aggregate consistency, duplicate IDs, and the SHA-256 manifest. The release benchmark check is provided by `scripts/validate_benchmark.py`.

The raw synthetic training pools are intentionally archived from the release.
The retained training validator can still inspect an externally restored pool;
it requires the pool directory explicitly and never assumes a release dataset.

The benchmark validator is the release check for the 48-case benchmark:

```bash
python3 scripts/validate_benchmark.py
```

The committed fixtures are intentionally small (3 cases, 3 dialogues, 1 exercise, and 1 evaluation result), so the balance flag is not used for them.
