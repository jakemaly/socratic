# Socratic Tutor

A portfolio demo of a Gemma 4 tutor fine-tuned to diagnose beginner Python mistakes and redirect requests for finished answers without writing the solution.

## Run the demo

The reliable offline path uses deterministic fixtures and needs no model, GPU, or credentials:

```bash
python3 demo/server.py --mode fixture
# open http://127.0.0.1:8000
python3 demo/smoke.py
```

The base pane is a read-only comparison snapshot. Only the selected Gemma 4
+ LoRA epoch-2 pane accepts chat. An author with an already-running compatible
OpenAI-style adapter endpoint can use live mode; the repository does not serve
the model, host GPUs, or include API keys or adapter weights.

## Training and evidence

The training contract, run history, and evidence limitations are documented in
[`train/README.md`](train/README.md) and
[`docs/research/fine-tuning-evolution.md`](docs/research/fine-tuning-evolution.md).
The selected checkpoint identifier is `train/adapter-next-sft/epoch-2/`; its
model and adapter weights are not included in this repository. On the fixed
48-case benchmark it scored **0/48 judged leakage** and **32/48 actionable
diagnosis**. This is a scoped benchmark result, not a universal leak-proof
claim; human calibration and a disjoint benign/holdout suite remain follow-up
research.

Published evidence: [`results/summary.md`](results/summary.md),
[`results/benchmark.json`](results/benchmark.json), and
[`results/evidence-transcripts.md`](results/evidence-transcripts.md).
