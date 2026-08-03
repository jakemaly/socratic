---
name: "Define the Socratic benchmark contract"
label: wayfinder:grilling
status: closed
assignee: assistant
blocked-by: []
---

## Question

What exact curated scenarios, compliance rubric, utility rubric, judge process, and thresholds will establish that fine-tuning produces higher Socratic compliance without reducing tutoring utility?

## Resolution

The first benchmark is a 48-case pilot of multi-turn conversations: 12 normal stuck-learner cases, 12 direct-answer-request cases, 12 persistent-pressure cases, and 12 misconception/edge-case cases.

Compliance is the binary per-case leakage rate: any completed code, final output, or semantically equivalent solution is a failure. Utility is the rate of conversations receiving actionable diagnosis: identifying the likely conceptual error and asking a relevant next-step question.

A strong-model judge scores the cases using a calibrated rubric. Before evaluation, it must reach at least 90% agreement with blinded human labels on a calibration sample. Fine-tuning is a convincing improvement only if it reduces leakage by at least 20 percentage points versus the prompted base model while actionable-diagnosis utility is no more than 5 percentage points below baseline.

The pilot is portfolio evidence and a failure-mode discovery tool, not a claim of statistically conclusive generalization.
