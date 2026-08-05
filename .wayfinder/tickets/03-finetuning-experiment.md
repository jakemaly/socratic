---
name: "Set the fine-tuning comparison"
label: wayfinder:grilling
status: closed
assignee:
blocked-by: ["Define the Socratic benchmark contract"]
---

## Question

Which existing base model, fine-tuning method, controls, training budget, and reproducibility artifacts are needed for a credible comparison against the prompted base model?

## Resolution

Fine-tune Qwen2.5-7B-Instruct with LoRA (rank 32, alpha 64, all linear layers, lr 2e-4 cosine, 3 epochs) on the 400 gated synthetic dialogues, trained locally on an RTX 3090 with a single pinned seed, evaluating the final checkpoint only (no eval-set-driven checkpoint selection).

The baseline is the same base model carrying the identical tutor system prompt, decoded greedily (temperature 0); both models face the same 48 cases judged by the same calibrated strong-model judge. This isolates the adapter's effect.

Reproducibility artifacts committed to the repo: pinned seed, full training config, LoRA adapter, dataset version, judge version, and eval logs. The pilot claim stays as the benchmark contract defines it: a 20-point leakage reduction with no more than a 5-point utility drop.
