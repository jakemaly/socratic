# Socratic Tutor

A reproducible fine-tuning and evaluation project for a Gemma 4 tutor that diagnoses beginner Python mistakes and redirects requests for finished answers without writing the solution.

## Evaluation

The selected checkpoint is `train/adapter-next-sft/epoch-2/`. On the fixed
48-case benchmark it scored **0/48 judged leakage** and **32/48 actionable
diagnosis**. This is a scoped benchmark result, not a universal leak-proof
claim; human calibration and a disjoint benign/holdout suite remain follow-up
research.

Run the standard-library validation checks from the repository root:

```bash
python3 scripts/validate_data.py
python3 scripts/validate_benchmark.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

The evaluation harness is documented in [`eval/README.md`](eval/README.md).
Training history and limitations are documented in
[`train/README.md`](train/README.md) and
[`docs/research/fine-tuning-evolution.md`](docs/research/fine-tuning-evolution.md).

Published evidence: [`results/summary.md`](results/summary.md),
[`results/benchmark.json`](results/benchmark.json), and
[`results/evidence-transcripts.md`](results/evidence-transcripts.md).
