# Demo frontend v1 implementation record

> **Status: complete.** The fixture-first demo, optional live-adapter seam,
> published evidence, responsive layout, and local acceptance checks are in the
> repository. This document preserves the original scope and acceptance
> criteria; it is not a deployment or model-serving plan.

## Goal

Ship a local, portfolio-ready demo for the six fixed gallery exercises. The first viewport shows the same learner moment in two panes:

- **Base model:** a clearly labeled, read-only comparison snapshot.
- **Fine-tuned Gemma 4:** the only live-chat pane; it accepts chat only when the author supplies an existing compatible endpoint.

The demo must make the behavioral difference legible in 20 seconds, then support a short follow-up conversation with the fine-tuned tutor.

## Scope

Included:

- Six existing gallery exercises from `content/exercises.json`.
- Three existing behavior chips from `content/chips.json`.
- Base-model comparison snapshots and fine-tuned starter transcripts.
- Fine-tuned chat: chips, free-text follow-ups, restart, loading, timeout, and retry.
- Evidence section for benchmark results and two annotated transcripts.
- Local fixture mode and local live-adapter mode.
- Keyboard-accessible, responsive layout.

Not included:

- Serving the base model.
- Streaming responses in v1.
- Accounts, hosting, rate limiting, arbitrary exercise authoring, or semantic scoring in the browser.

## Proposed files

```text
demo/
  index.html       static app shell
  app.js           DOM rendering, interactions, adapter calls
  state.mjs        pure conversation state reducer
  styles.css       responsive visual treatment
  server.py        static server and same-origin /api/chat proxy
  fixtures.json    six snapshots/starters and offline replies
  smoke.py         scripted fixture-mode acceptance check
  README.md        local run instructions
```

`app.js` should load the existing `content/exercises.json` and `content/chips.json` rather than duplicating gallery content.

## Interaction model

1. Select an exercise from the fixed gallery.
2. Render its prepared base snapshot and fine-tuned starter state immediately.
3. Chips append a learner message to the fine-tuned history only.
4. Free-text input appends a learner message to the fine-tuned history only.
5. The base pane never changes during a live conversation.
6. Restart clears the fine-tuned history and restores the selected exercise's prepared states.
7. Changing exercises aborts or invalidates any pending request and restores that exercise.

The UI must label the panes explicitly as `Comparison snapshot` and `Live fine-tuned tutor`; never imply that both models are being served.

## API contract

The browser calls one same-origin endpoint:

```http
POST /api/chat
Content-Type: application/json

{
  "exercise_id": "exercise-gallery-even-or-odd",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0
}
```

Response:

```json
{"message": {"role": "assistant", "content": "..."}}
```

`demo/server.py` keeps credentials server-side and forwards to the existing OpenAI-compatible client shape. It serves one configured adapter target using `SOCRATIC_API_BASE`, `SOCRATIC_API_KEY`, and `SOCRATIC_ADAPTER_MODEL`. Fixture mode must work without credentials.

No client-side model selector is needed. The server should reject malformed payloads and cap request size before forwarding.

## Evidence contract

The benchmark track publishes these small GitHub artifacts together:

- `results/benchmark.json` with the selected epoch-2 comparison and full per-case scores;
- `results/summary.md` with the base and epoch 2/3/4 table, methodology, and limitations;
- `results/evidence-transcripts.md` with two annotated epoch-2 transcripts;
- `results/demo-evidence.json`, the presentation record consumed by the demo.

The presentation record identifies `train/adapter-next-sft/epoch-2` as the
selected checkpoint and contains counts and rates for the fixed 48-case
benchmark: especially **0/48 judged leakage** and **32/48 actionable diagnosis**.
It also includes the benchmark/judge metadata, the base comparison, diagnosis
and leakage deltas, two transcript kinds (`actionable-diagnosis` and
`firm-answer-redirect`), limitations, and source links to the three result
artifacts.

No promotion thresholds are applied to this presentation result. The portfolio
claim is scoped to the fixed benchmark and must not imply universal leak-proof
behavior. Calibration is represented as `not_performed` or
`unavailable` unless a real human calibration file exists; never invent an
agreement percentage.

`schemas/demo_evidence.schema.json` is the authoritative shape for
`results/demo-evidence.json`, and the browser independently checks required
completeness before replacing the unavailable state. Avoid adding a Markdown
parser dependency. Until a complete record is published, render an explicit
unavailable state rather than placeholder metrics.

## Build order (completed)

1. **Fixture shell:** static page, gallery selection, two-pane layout, prepared snapshots, tuned starter messages.
2. **Interactions:** chips, fine-tuned composer, follow-up history, restart, exercise changes, stale-request guard.
3. **Local server:** same-origin static server, fixture mode, one-adapter `/api/chat` proxy, validation and error states.
4. **Evidence:** publish the real epoch-2 presentation record and link to the source artifacts.
5. **Polish:** responsive/mobile layout, keyboard and focus checks, reduced-motion behavior, copy and spacing pass.
6. **Demo check:** run the scripted 20-second path; recording is an optional presentation aid, not a release artifact.

Steps 1, 2, 4, and 5 are implemented and covered by the local checks. Step 3
remains an optional author-run path using an existing compatible endpoint; the
repository does not provide an adapter server. Step 6 is represented by
`python3 demo/smoke.py` and the documented manual presentation path; recording
is optional and no hosted release is part of this project.

## Acceptance checks

- The first viewport shows both model names, the same learner moment, and which pane is live.
- The base pane is never sent a request and never accepts input.
- Every chip and free-text submission affects only the fine-tuned history.
- Restart and exercise changes cannot be overwritten by an older response.
- An adapter timeout leaves the base snapshot visible and offers retry.
- Fixture mode works offline and has no completed code in the fine-tuned tutor replies.
- Evidence shows real counts/rates for the fixed benchmark or a clear unavailable state; no invented metrics.
- Evidence identifies epoch 2, shows 0/48 leakage and 32/48 diagnosis, and states the benchmark limitation.
- Calibration is explicitly not performed/unavailable unless backed by a human calibration file.
- Layout remains usable on mobile, with visible focus and readable contrast.
- `python3 demo/smoke.py` passes from the repository root.

## 20-second recording path

1. Select **Even-or-odd**.
2. Show the static base leak beside the fine-tuned redirect.
3. Click **just give me the answer**.
4. Type **What should I check first?** in the fine-tuned composer.
5. Point to **Evidence** and the benchmark comparison.
