# Gemma 4 31B QLoRA run

`gemma4_qlora_guided.ipynb` is the canonical training recipe. It follows
[Unsloth's Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train) because
the generic Transformers/PEFT path is not the supported path for this model.

## Fixed experiment

- **Model:** `unsloth/gemma-4-31B-it-unsloth-bnb-4bit`, revision
  `8e256fc6d63003fc0ca8c91b976e6dcc38433385`
- **Loader:** Unsloth `FastModel`, pre-quantized 4-bit QLoRA, no CPU offload
- **GPU target:** one RTX 3090 with 24 GiB VRAM
- **Memory settings:** max sequence length 1024, micro-batch 1, gradient
  accumulation 4, Unsloth gradient checkpointing
- **Adapter:** language + attention + MLP modules, rank 16, alpha 32, dropout 0
- **Training:** three epochs, learning rate `2e-4`, cosine schedule, `adamw_8bit`,
  assistant responses only, one seed, final adapter only

Unsloth documents approximately 22 GB VRAM for 31B QLoRA. The pre-quantized
checkpoint is about 19 GB on disk. This machine has 24 GB VRAM and 31.5 GB system
RAM, so the recipe intentionally avoids CPU offload
and starts with the 1,024-token smoke test. The machine currently has an 8 GB
swapfile; swap is not a training memory strategy and should not be relied on.

## Environment

Use a clean Python 3.12 environment. Do not retrofit the old Qwen environment:
Gemma 4 requires the pinned newer Transformers/TRL versions and Unsloth.

```bash
nvidia-smi
uv venv .venv-train --python 3.12
source .venv-train/bin/activate
uv pip install -r train/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
python -m ipykernel install --user --name socratic-train \
  --display-name "Python 3 (socratic training)"
jupyter lab
```

Select the `socratic-train` kernel and open the notebook from the repository
root. Gemma model access may require accepting Google's terms on Hugging Face.
The notebook prints the actual GPU, RAM, package versions, and peak VRAM.

## Run order

1. Leave `RUN_MODE = "smoke"` and run cells top-to-bottom.
2. Stop if the hardware, dependency, formatting, response-mask, loss, or memory
   gate fails. Read the printed base replies and peak memory.
3. Only after smoke succeeds, restart the kernel, set `RUN_MODE = "full"` and
   `CONFIRM_FULL_RUN = True`, then run top-to-bottom again.
4. The full run writes `train/adapter/`, `train/config.yaml`, `train/seed`,
   `train/logs/`, and `train/adapter.sha256`.
5. Evaluate the saved adapter separately with the fixed 48-case benchmark and
   compare leakage and actionable diagnosis against the same Gemma base.

The old `qwen_lora_guided.ipynb` remains untouched because it contained
uncommitted work; it is no longer the canonical recipe.

## V2 run

`gemma4_qlora_v2.ipynb` uses the separate `data/train-v2/` pool and writes to
`train/adapter-v2/`, `train/config-v2.yaml`, `train/logs-v2/`, and
`train/adapter-v2.sha256`; it never overwrites the V1 adapter. The notebook
defaults to smoke mode. For a confirmed full run, execute it with
`SOCRATIC_RUN_MODE=full SOCRATIC_CONFIRM_FULL_RUN=true` after the smoke gate
passes, then benchmark the saved adapter using the fixed 48-case evaluation.
