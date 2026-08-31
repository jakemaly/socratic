# Demo frontend v1 implementation plan

## Goal

Ship a local, portfolio-ready demo for the six fixed gallery exercises. The first viewport shows the same learner moment in two panes:

- **Base model:** a clearly labeled, read-only comparison snapshot.
- **Fine-tuned Gemma 4:** the only served model and the only pane that accepts chat input.

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

Issue #16 must publish these artifacts together:

- `results/benchmark.json` with full per-case scores;
- `results/summary.md` with the comparison table and verdict;
- `results/evidence-transcripts.md` with two annotated transcripts;
- the calibration agreement log;
- `results/demo-evidence.json`, the presentation record consumed by the demo.

The presentation record must derive from those #16 outputs and contain:

- base vs fine-tuned leakage rate;
- base vs fine-tuned actionable-diagnosis rate;
- 20-point leakage and 5-point utility thresholds;
- calibration agreement;
- one actionable-diagnosis transcript;
- one firm-answer-redirect transcript;
- the explicit pass/fail verdict;
- `source_artifacts` links to `benchmark.json`, `summary.md`, and
  `evidence-transcripts.md`.

`schemas/demo_evidence.schema.json` is the authoritative shape for
`results/demo-evidence.json`. Rates and `calibration_agreement` are either
numbers from 0 through 1 or percent strings; deltas and thresholds are numeric
percentage points. `thresholds` contains `leakage_reduction_points` and
`utility_drop_points`. `transcripts` contains exactly two records, each with a
non-empty `title`, `annotation`, and at least two `{role, content}` messages.
Every `source_artifacts` entry has a non-empty `label` and `path`, and the list
links to `benchmark.json`, `summary.md`, and `evidence-transcripts.md`.

Issue #16 must validate the record against that schema before publishing it.
The browser independently checks required completeness before replacing the
unavailable state. Avoid adding a Markdown parser dependency. Until #16
publishes this complete artifact set, render an explicit unavailable state
rather than placeholder metrics.

## Build order

1. **Fixture shell:** static page, gallery selection, two-pane layout, prepared snapshots, tuned starter messages.
2. **Interactions:** chips, fine-tuned composer, follow-up history, restart, exercise changes, stale-request guard.
3. **Local server:** same-origin static server, fixture mode, one-adapter `/api/chat` proxy, validation and error states.
4. **Evidence:** wire the real #16 presentation record and link to the source artifacts.
5. **Polish:** responsive/mobile layout, keyboard and focus checks, reduced-motion behavior, copy and spacing pass.
6. **Demo check:** run the scripted 20-second path and record the final video.

Steps 1–2 are unblocked now. Step 3 needs the adapter serving artifact from #11. Step 4 needs the benchmark evidence from #16.

## Acceptance checks

- The first viewport shows both model names, the same learner moment, and which pane is live.
- The base pane is never sent a request and never accepts input.
- Every chip and free-text submission affects only the fine-tuned history.
- Restart and exercise changes cannot be overwritten by an older response.
- An adapter timeout leaves the base snapshot visible and offers retry.
- Fixture mode works offline and has no completed code in the fine-tuned tutor replies.
- Evidence shows real numbers or a clear unavailable state; no invented metrics.
- Layout remains usable on mobile, with visible focus and readable contrast.
- `python3 demo/smoke.py` passes from the repository root.

## 20-second recording path

1. Select **Even-or-odd**.
2. Show the static base leak beside the fine-tuned redirect.
3. Click **just give me the answer**.
4. Type **What should I check first?** in the fine-tuned composer.
5. Point to **Evidence** and the benchmark comparison.
