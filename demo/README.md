# Socratic Tutor demo

A local, fixture-first portfolio demo for six fixed beginner-Python exercises.
The base model is a read-only comparison snapshot. Only the selected fine-tuned
Gemma 4 epoch-2 pane accepts chat.

## Fixture mode

No model, GPU, or credentials are required:

```bash
python3 demo/server.py --mode fixture
```

Open <http://127.0.0.1:8000>. Run the offline acceptance check from the
repository root with `python3 demo/smoke.py`.

## Live adapter mode

Live mode is an optional author-run path. The author must provide an already
running OpenAI-compatible `/chat/completions` endpoint; this repository does
not provide model serving, public hosting, auth, or GPU infrastructure:

```bash
export SOCRATIC_DEMO_MODE=live
export SOCRATIC_API_BASE=http://127.0.0.1:YOUR_PORT/v1
export SOCRATIC_API_KEY=local-development-key
export SOCRATIC_ADAPTER_MODEL=your-gemma4-adapter
python3 demo/server.py --mode live
```

Credentials stay server-side. The server prepends the canonical tutor prompt
from `eval/judge.py` when needed, forwards only the fine-tuned adapter request,
and leaves the base pane static. Model weights are not included in Git.

## Frontend contract

- `content/exercises.json` and `content/chips.json` are the source of truth for
  gallery content and behavior chips.
- `demo/fixtures.json` supplies deterministic comparison snapshots and tuned
  starter/reply content.
- `POST /api/chat` accepts an `exercise_id`, message history, and temperature,
  and returns `{ "message": { "role": "assistant", "content": "..." } }`.
- Evidence is loaded from `results/demo-evidence.json` only after it passes the
  completeness checks in `demo/evidence.mjs` and
  `schemas/demo_evidence.schema.json`; the UI never invents metrics or parses
  Markdown.

The published claim is limited to the fixed 48-case benchmark: epoch 2 scored
0/48 judged leakage and 32/48 actionable diagnosis. Calibration is explicitly
not performed for this result, and no universal leak-proof claim is made.
See [`docs/research/fine-tuning-evolution.md`](../docs/research/fine-tuning-evolution.md)
for the detailed training history.
