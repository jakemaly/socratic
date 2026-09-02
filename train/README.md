# Training record

This release does not ship raw training pools, notebooks, model weights, or
training environments. Those materials are intentionally archived in Git
history rather than included in the repository.

## Selected experiment

The published result comes from a Gemma 4 31B instruction-tuned model with
pre-quantized 4-bit QLoRA, `all-linear` rank-16 adapters, assistant-only loss,
and a fixed 48-case evaluation benchmark. The four-epoch run compared snapshots
2, 3, and 4; epoch 2 is selected because it is the earliest snapshot with zero
judged leakage.

- Leakage: **0/48 judged leakage**
- Actionable diagnosis: **32/48**
- Scope: fixed benchmark and semantic judge, not a universal safety claim

Read the complete methodology and comparison in
[`docs/research/fine-tuning-evolution.md`](../docs/research/fine-tuning-evolution.md)
and the published evidence in [`results/summary.md`](../results/summary.md).

## Retained receipts

- `baseline.yaml` records the prompted-base model contract used for comparison.
- `config-next-sft.yaml` records the selected run's resolved model, data hashes,
  hyperparameters, and runtime. Its dataset path identifies an archived input;
  it is a receipt, not a runnable configuration in this release.
- `scripts/training_contract.py`, `scripts/gates.py`,
  `scripts/generate_dialogues.py`, and `scripts/validate_train.py` remain as
  inspectable implementation history. The generator requires an explicit output
  directory so it cannot silently recreate a release dataset.

The selected model and adapter weights are not included in Git. The retained
receipts and evaluation artifacts document the run without requiring a serving
layer.
