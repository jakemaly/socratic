---
name: "Shape the fixed exercise gallery"
label: wayfinder:grilling
status: closed
assignee:
blocked-by: ["Define the Socratic benchmark contract", "Set the fine-tuning comparison"]
---

## Question

What fixed beginner-Python exercises and presentation flow will let visitors experience the tutor while seeing enough benchmark evidence to assess compliance, utility, and the effect of fine-tuning?

## Resolution

A fixed gallery of six classic intro-Python exercises, one per core topic: Greeting (variables/I-O), Even-or-odd (conditionals), Countdown (loops), Average of a list (lists), Temperature converter (functions), and Word counter (strings/dicts).

The gallery exercises come from a third pool, distinct from both the 48 benchmark cases and the 400 training-pool dialogues, keeping the benchmark instrument clean and the gallery publishable openly.

The live chat runs the fine-tuned model only (no base-model toggle mid-session). The chat UI offers suggestion chips — "I'm stuck", "just give me the answer", "hint" — that surface the tutor's redirect behavior, plus a restart button.

Benchmark evidence lives in a dedicated Evidence section, separate from the tutoring flow: a compliance/utility table (base vs fine-tuned, with the 20-point leakage and 5-point utility thresholds called out), the strong-model judge's calibration statistic, and two annotated transcripts (one showing good actionable diagnosis, one showing a firm redirect of an answer demand).
