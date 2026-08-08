# Guided Qwen2.5-7B LoRA run

`qwen_lora_guided.ipynb` is the reproducible, human-run training path for the
400-dialogue pilot. It is deliberately a lesson and a runbook: each section has a
checkpoint that either proves a premise or stops before an expensive step.

## Fixed experiment

- Base: `Qwen/Qwen2.5-7B-Instruct`, Hub revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Data: `data/train/dialogues.jsonl`, exactly 400 gated records; the notebook
  checks `data/train/manifest.sha256` and runs `scripts/validate_train.py`.
- Adapter: PEFT LoRA, `r=32`, `lora_alpha=64`, `target_modules="all-linear"`,
  dropout `0`, bias `none`.
- Training: BF16, 3 epochs, learning rate `2e-4`, cosine schedule, batch `1`,
  gradient accumulation `1`, max length `1024`, no packing, one seed, final
  adapter only.
- Loss: assistant responses only. Qwen 2.5's Hub template does not expose TRL's
  assistant-token markers, so the notebook installs a deterministic equivalent
  template and proves its mask before constructing the trainer.
- Benchmark scoring is separate. `baseline.yaml` records the prompted-base arm:
  the same base revision, tutor prompt, and greedy decoding (`temperature: 0`).

The initial Microsoft [`microsoft/LoRA`](https://github.com/microsoft/LORA)
repository was considered, but the agreed implementation uses the current Hugging
Face PEFT/TRL stack. Microsoft's low-level `loralib` is not installed or vendored;
the link is background, not a second training implementation.

## Environment

Use a clean environment, not an application environment:

```bash
nvidia-smi
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
uv pip install -r train/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m ipykernel install --user --name socratic-train --display-name "Python 3 (socratic training)"
jupyter lab
```

The PyTorch wheel must match the machine's driver. If the official selector gives
a different CUDA wheel, record that exact change in the run's environment report;
do not silently call it the same run. The model is public, so a Hugging Face token
is not required.

## Notebook order

1. Select the `socratic-train` kernel and run from the repository root.
2. Leave `RUN_MODE = "smoke"` for the first pass. This runs the data audit, a
   download-free CPU PEFT self-check, tokenizer/mask checks, a three-case base
   preflight, and one real GPU optimizer step.
3. Read every assertion and inspect the printed loss, trainable-parameter list,
   token mask, and base-model replies. A failed gate is a stop, not a reason to
   change hyperparameters.
4. To run the fixed job, restart the kernel to release the preflight model, then
   set `RUN_MODE = "full"` and `CONFIRM_FULL_RUN = True` in the setup cell. Run
   top-to-bottom again. The full path refuses to reuse a non-empty output folder.
5. The full path saves `train/adapter/`, `train/adapter.sha256`,
   `train/config.yaml`, `train/seed`, and `train/logs/`.

The smoke path writes its adapter to a temporary directory and deletes it. It does
not create a fake final result. This repository change adds the gates but does not
claim that the GPU smoke or full job has passed; execute them on the training
machine before reporting a run.

## Reproducibility boundary

The seed, data seed, model revision, dataset hash, chat-template hash, package
versions, GPU/CUDA report, effective training arguments, and log history are
written with a full run. CUDA kernels and fused operations can still prevent
bit-for-bit identical loss values across machines; the artifact records the
conditions needed to explain any small drift. There is no hyperparameter search,
multiple-seed claim, checkpoint selection, or QLoRA fallback in this recipe.

After training, run the separate benchmark using `baseline.yaml` and the same
revision, tutor prompt, cases, decoding, and judge for both arms.
