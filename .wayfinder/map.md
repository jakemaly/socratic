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
- Tracker: local Markdown; ticket files are the child issues, and `blocked-by` records dependencies.

## Decisions so far

- Destination and scope — this map finds a portfolio-quality tutor demo plus comparative benchmark evidence.
- Learner and subject — adult beginners learning introductory Python.
- Tutor contract — questions, feedback, diagnosis, and resources are allowed; completed or semantically equivalent solutions are not.
- Initial benchmark posture — curated conversations, compared against the prompted base model.
- Data direction — synthetic dialogues are the initial fine-tuning data source.
- Success claim — materially higher compliance with equal tutoring utility.
- Demo boundary — a fixed exercise gallery, not an unbounded free-form product.

## Not yet specified

- The exact compliance and utility rubric, scenario set, scoring method, and pass thresholds.
- How synthetic conversations are generated, filtered, audited for leakage, and versioned.
- Which base model, fine-tuning method, training budget, and reproducibility level make the comparison credible.
- The minimum demo/evaluation artifact needed to present results clearly and safely.
- Later implementation details: dataset schema, evaluator implementation, serving stack, and exercise authoring process.

## Out of scope

- A general-purpose educational platform.
- Broad multi-subject tutoring.
- Pretraining a foundation model.
- Proving real-world learning gains with a large learner study in the first iteration.
- An unrestricted public chatbot.
