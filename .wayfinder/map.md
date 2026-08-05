---
name: "Socratic Tutor Portfolio Map"
label: wayfinder:map
status: open
---

## Destination

A working tutor demo for adult beginners learning introductory Python, backed by benchmark evidence that fine-tuning improves strict Socratic behavior over a prompted base model without reducing tutoring utility.

The tutor asks guiding questions, diagnoses missing logic, points to learning resources, and firmly redirects answer demands. It never gives completed code, final output, or an equivalent prose solution.

## Notes

- Domain: introductory Python tutoring and AI alignment/fine-tuning portfolio work.
- Skills: grilling, domain-modeling, research, prototype, and tdd as decisions become actionable.
- Standing preference: keep the first version narrow, inspectable, and benchmarkable; prefer evidence over platform breadth.
- Tracker: GitHub Issues; [Socratic Tutor Portfolio Map](https://github.com/jakemaly/socratic/issues/1) is canonical, with GitHub sub-issues for child tickets and `Blocked by` body lines for sequencing.

## Decisions so far

- Destination and scope — this map finds a portfolio-quality tutor demo plus comparative benchmark evidence.
- Learner and subject — adult beginners learning introductory Python.
- Tutor contract — questions, feedback, diagnosis, and resources are allowed; completed or semantically equivalent solutions are not.
- Initial benchmark posture — curated conversations, compared against the prompted base model.
- Data direction — synthetic dialogues are the initial fine-tuning data source.
- Success claim — materially higher compliance with equal tutoring utility.
- Demo boundary — a fixed exercise gallery, not an unbounded free-form product.
- [Define the Socratic benchmark contract](https://github.com/jakemaly/socratic/issues/2) — 48 multi-turn cases; binary leakage and actionable diagnosis; calibrated strong-model judging; 20-point compliance gain with a 5-point utility floor.
- [Choose the synthetic dialogue pipeline](https://github.com/jakemaly/socratic/issues/3) — 400 balanced synthetic dialogues from a separate exercise pool, filtered by pinned compliance and utility gates with rejection provenance logged.
- [Set the fine-tuning comparison](https://github.com/jakemaly/socratic/issues/4) — Qwen2.5-7B-Instruct with LoRA (rank 32, lr 2e-4, 3 epochs) on the 400 dialogues; prompted-base baseline at temperature 0; single pinned seed, final checkpoint, config/adapter/logs committed.
- [Shape the fixed exercise gallery](https://github.com/jakemaly/socratic/issues/5) — six classic exercises (variables/I-O, conditionals, loops, lists, functions, strings/dicts) from a third pool; fine-tuned model live with chips + restart; dedicated Evidence section (compliance/utility table, judge calibration, two annotated transcripts).

## Implementation frontier

The decision phase is complete — all four grilling tickets are closed. Implementation is specced in 13 child tickets split by executor; the full specs (goal, spec, deliverables, acceptance criteria) live in each ticket's body.

**Agent tracks** (labeled `ready-for-agent`):
- [#6](https://github.com/jakemaly/socratic/issues/6) Dataset schema & fixtures — canonical versioned schemas + fixtures; unblocks all tracks.
- [#7](https://github.com/jakemaly/socratic/issues/7) Author the 48 benchmark cases — 4 families × 12, third exercise pool.
- [#8](https://github.com/jakemaly/socratic/issues/8) Evaluator + calibrated judge harness — scoring, judge client, calibration export.
- [#9](https://github.com/jakemaly/socratic/issues/9) Synthetic dialogue pipeline — 400 gated dialogues with rejection provenance.
- [#11](https://github.com/jakemaly/socratic/issues/11) Fine-tuning run — Qwen2.5-7B LoRA per the fine-tuning comparison decision.
- [#13](https://github.com/jakemaly/socratic/issues/13) Gallery content pack — six exercises + chips copy.
- [#16](https://github.com/jakemaly/socratic/issues/16) Benchmark run + evidence pack — calibration gate, both arms, verdict, 2 annotated transcripts.
- [#17](https://github.com/jakemaly/socratic/issues/17) Demo app — chat + chips + restart + Evidence section.

**Human gates** (labeled `ready-for-human`):
- [#10](https://github.com/jakemaly/socratic/issues/10) Infra & credentials — API key, GPU check; parallel provisioning.
- [#12](https://github.com/jakemaly/socratic/issues/12) Approve the 48 benchmark cases — review of #7.
- [#14](https://github.com/jakemaly/socratic/issues/14) Approve gallery content — review of #13.
- [#15](https://github.com/jakemaly/socratic/issues/15) Blinded calibration labels — the one hard human gate (15–20 transcripts, blocks #16).
- [#18](https://github.com/jakemaly/socratic/issues/18) Sign off on the claim — verdict on #16's numbers.

Parallelism: after #6, four agent tracks (#7, #8, #9, #13) run concurrently with #10 provisioning; #11 follows #9 + #10; #16 follows #7 + #8 + #11 + #15; #17 scaffolds against stubs and wires #11 + #13 + #16 last. Only #16 waits on a human deliverable (#15).

## Out of scope

- A general-purpose educational platform.
- Broad multi-subject tutoring.
- Pretraining a foundation model.
- Proving real-world learning gains with a large learner study in the first iteration.
- An unrestricted public chatbot.
