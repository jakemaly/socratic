# Socratic Tutor demo

A local, fixture-first portfolio demo for the six fixed beginner-Python exercises.
The base model is a read-only comparison snapshot. Only the fine-tuned Gemma 4
adapter accepts chat messages, so the demo fits on a single GPU.

## Fixture mode

No model or credentials are required:

```bash
python3 demo/server.py --mode fixture
```

Open <http://127.0.0.1:8000>.

Run the offline smoke check from the repository root:

```bash
python3 demo/smoke.py
```

## Live adapter mode

The adapter must be available through an OpenAI-compatible `/chat/completions`
endpoint. Keep credentials in the environment; they are never sent to the
browser:

```bash
export SOCRATIC_DEMO_MODE=live
export SOCRATIC_API_BASE=http://127.0.0.1:YOUR_PORT/v1
export SOCRATIC_API_KEY=local-development-key
export SOCRATIC_ADAPTER_MODEL=your-gemma4-adapter
python3 demo/server.py --mode live
```

The server prepends the canonical tutor prompt from `eval/judge.py` when the
browser request does not include a system message. The live endpoint receives
only the fine-tuned adapter request; the base pane remains static.

## Frontend contract

- `content/exercises.json` and `content/chips.json` are the source of truth for
  gallery content and behavior chips.
- `demo/fixtures.json` supplies deterministic comparison snapshots and tuned
  starter/reply content.
- `POST /api/chat` accepts an `exercise_id`, message history, and temperature,
  and returns `{ "message": { "role": "assistant", "content": "..." } }`.
- Issue #16 must publish `results/demo-evidence.json` alongside
  `results/benchmark.json`, `results/summary.md`, and
  `results/evidence-transcripts.md`; the UI remains unavailable until that
  presentation record exists and does not invent metrics or parse Markdown.
