# Qwen2.5-7B LoRA training stack for an RTX 3090

Research for issue #11 (2026-08-07). This note compares practical software stacks for the repository's fixed run: `Qwen/Qwen2.5-7B-Instruct`, LoRA rank 32 / alpha 64, all linear layers, learning rate `2e-4`, cosine schedule, three epochs, one pinned seed, and final checkpoint only.

## Short answer

Use a dedicated Python environment with **PyTorch CUDA + Hugging Face Transformers + PEFT + TRL's `SFTTrainer` + Accelerate**. Start with ordinary BF16 LoRA because the repository's fixed decision says LoRA and the records are short; use bitsandbytes 4-bit QLoRA only as an explicitly recorded fallback if the real 24 GiB run does not fit. Do not make Unsloth, Axolotl, or torchtune the canonical run unless the smoke test shows the standard stack cannot complete it.

This choice keeps the data path, Qwen chat template, adapter format, benchmark loader, and training configuration in the same Hugging Face ecosystem. It also avoids hiding the effective recipe behind a launcher or optimized fork.

## Constraints from the repository

- The training set is exactly 400 gated conversations in `data/train/dialogues.jsonl`; its manifest records SHA-256 hashes for the JSONL artifacts.
- The repository currently has no training dependency lockfile, trainer, or adapter artifacts. The first implementation must therefore pin the environment and record the resolved package versions, CUDA/runtime details, model revision, dataset hashes, and complete effective config.
- The existing dataset records contain `turns` with `system`, `user`, and `assistant` messages. The Qwen tokenizer's chat template must be used rather than hand-written ChatML. The model card demonstrates `tokenizer.apply_chat_template(...)` and says Transformers `>=4.37.0` is required.
- The benchmark's canonical baseline prompt is `eval/judge.py:TUTOR_SYSTEM_PROMPT`; `train/baseline.yaml` should quote it verbatim and use temperature `0`.

## Hardware reality

Qwen's model card reports 7.61B parameters and a 32,768-token configuration (with optional YaRN for longer contexts). A 7B model in 16-bit weights alone is roughly 15 GB before activations, gradients, optimizer state, and CUDA workspace. Full fine-tuning is therefore not a realistic 24 GiB target. Adapter training is realistic; 4-bit QLoRA leaves substantially more activation headroom than 16-bit LoRA.

The 3090 is an Ampere 24 GiB card. Whether a particular PyTorch/attention/quantization build works efficiently is an installation concern, not something the repository can infer. The smoke test must measure peak allocated/reserved memory with the actual sequence lengths and batch settings.

## Stack comparison

| Stack | What it provides | Fit for this run | Main risk |
|---|---|---|---|
| **Transformers + PEFT + TRL + bitsandbytes** | First-party Hugging Face model loading, PEFT LoRA, TRL `SFTTrainer`, and 4-bit loading through `BitsAndBytesConfig`. TRL accepts conversational `messages` data and applies the tokenizer chat template. | **Best default.** Smallest conceptual gap from the existing JSONL and easiest to audit: explicit Python config, standard adapter format, and direct control over every issue #11 setting. | More glue code and more dependency/version interactions than a turnkey launcher. QLoRA-specific settings and loss masking need an explicit smoke test. |
| **torchtune** | PyTorch-native recipes/configs for LoRA and QLoRA, memory optimizations, and reproducible YAML-driven runs. | Viable alternative: the current repository includes a Qwen2.5 7B single-device LoRA config. | Its recipe defaults differ from issue #11 (for example, rank 8/alpha 16 and a specific attention/MLP target policy); custom `turns` data and the requested all-linear mapping still need verification. |
| **Unsloth** | Optimized training wrappers, LoRA/QLoRA modes, notebooks, and lower-memory kernels; official docs recommend QLoRA for accessible hardware and document `load_in_4bit`, sequence length, batch, accumulation, and learning-rate controls. | Good fallback when the standard stack OOMs or throughput is unacceptable. It is likely the quickest path to a working single-GPU run. | Optimization layer and rapidly changing install/model variants complicate exact reproducibility. Do not silently substitute an Unsloth-quantized model for the official Qwen checkpoint; record the exact model artifact and versions. |
| **Axolotl** | YAML launcher supporting LoRA/QLoRA, local JSONL, preprocessing, training, inference, and adapter merge. | Practical if a declarative config and low implementation effort matter more than keeping the training path in repository Python. | Its dataset format/config conventions may require conversion from `turns`; defaults can hide effective settings. A generated/prepared dataset and all resolved config must be retained for auditability. |

### Transformers + PEFT + TRL

TRL's official SFT documentation states that conversational datasets use structured messages and that the trainer automatically applies the chat template. It exposes `peft_config`, `quantization_config`, `eos_token`, `max_length`, packing, completion-only loss, assistant-only loss, save strategy, seed, and scheduler settings. It explicitly documents combining a bitsandbytes quantization config with PEFT for QLoRA.

For this dataset, the least surprising representation is one record with `messages` equal to the existing `turns` (or a deterministic conversion that preserves every role/content string). Set the Qwen end-of-turn token explicitly if the installed TRL version does not infer it; TRL's docs call out `<|im_end|>` for Qwen2.5. Decide and record whether loss is on the whole formatted sequence or assistant messages only. That choice changes the experiment and must not be left to a library default.

PEFT's LoRA configuration provides `r`, `lora_alpha`, `target_modules`, dropout, bias, and related controls. “All linear layers” should be represented explicitly (for example, PEFT's supported all-linear target convention in the pinned version) and verified by printing the trainable module names/count; do not assume a shorthand has identical behavior across PEFT versions.

### QLoRA and bitsandbytes

QLoRA is the sensible memory-first option on 24 GiB: load the frozen base in 4-bit and train the LoRA parameters. Use bitsandbytes' official Transformers integration, pin the quantization type/compute dtype/double-quantization choices, and log them. Those settings are part of the recipe even though issue #11 only fixes the LoRA hyperparameters.

The repository decision says “LoRA” rather than explicitly “QLoRA.” Therefore the implementation must make the choice visible in `train/config.yaml`, not silently change it. A 16-bit LoRA run may fit at short sequence lengths with checkpointing and tiny micro-batches; QLoRA is the safer first smoke test for a 24 GiB card. If the intended evidence is specifically non-quantized LoRA, run that exact mode and treat QLoRA only as a documented fallback—not as the reported experiment.

## Recommended implementation

Use **Transformers + PEFT + TRL**, with bitsandbytes QLoRA only if the specified LoRA precision does not fit. Run it in a dedicated `uv`/virtualenv environment; do not reuse an unrelated application environment. The practical package boundary is:

- `torch` with a CUDA build compatible with the installed NVIDIA driver;
- `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `safetensors`, and `huggingface_hub`;
- `bitsandbytes` only for the explicitly recorded QLoRA fallback;
- no `flash-attn`, W&B, or UI launcher until a smoke test proves it is needed.

Pin the exact resolved versions after installing the CUDA build, then save `pip freeze`/`uv pip freeze` and the GPU/CUDA report with the run. The version numbers in upstream docs move quickly; a tested lock is more valuable here than an untested version copied from a blog.

Reasons:

1. It directly supports Qwen's native tokenizer chat template and the repository's conversational records.
2. PEFT exposes the exact requested rank, alpha, and target-module settings.
3. TRL provides a conventional SFT loop, cosine scheduler, logging, seed, and final adapter save without introducing a large launcher abstraction.
4. The resulting adapter is reloadable with standard Transformers/PEFT tooling and straightforward to evaluate through the existing `eval` harness.
5. A small repository-owned conversion function can map `turns` to TRL's `messages` field; a custom PyTorch training loop would add more code and more places for masking, scheduler, and checkpoint bugs.

The configuration should explicitly include at least:

- base model ID and immutable Hub revision;
- dataset paths and every dataset hash from `data/train/manifest.sha256`;
- schema/generator version and record count;
- seed and data seed;
- LoRA `r=32`, `alpha=64`, target-module policy (“all linear”), dropout, and bias policy;
- precision/mode (`fp16` LoRA or 4-bit QLoRA), quantization type/compute dtype/double quantization when applicable;
- learning rate `2e-4`, cosine scheduler, warmup policy, weight decay, optimizer;
- epochs `3`, micro-batch size, gradient accumulation, effective batch size, max sequence length, packing, and loss masking;
- gradient checkpointing and attention implementation;
- `save_strategy: final`/equivalent, output paths, and no checkpoint selection;
- exact package, Python, PyTorch, CUDA, GPU, and model revision metadata.

Keep the adapter output in PEFT `save_pretrained` format. Generate a sorted SHA-256 manifest covering every adapter file, plus the config, seed, logs, and environment report needed to reproduce it. The baseline config should use the same base-model revision and tokenizer/chat template but no adapter, the canonical tutor prompt verbatim, temperature `0`, and the same evaluation seed/cases when #16 is run.

## Smoke-test plan and unknowns

Before the full three-epoch run:

1. Verify the GPU, CUDA, PyTorch, Transformers, PEFT, TRL, bitsandbytes, and attention backend versions.
2. Load the exact model revision and run the Qwen chat-template round trip on one training record; assert the rendered text contains the expected role boundaries and `<|im_end|>` termination.
3. Convert one record to the trainer format and print token length plus assistant/loss masks. Confirm no turn is dropped and no system prompt is accidentally replaced.
4. Instantiate LoRA and print trainable parameter names/count. Confirm all intended linear projections are covered and the base parameters are frozen.
5. Run one optimizer step on one or two examples with the proposed sequence length, precision, batch, gradient checkpointing, and quantization mode; capture peak VRAM and check for NaN/Inf.
6. Run a short deterministic canary twice with the same seed and compare logged step/ loss behavior. Exact bitwise determinism may not hold with fused kernels; record that result rather than claiming it.
7. Save and reload the adapter on the base model, then generate one response using the exact baseline prompt and temperature. Confirm the adapter path and tokenizer work independently of the trainer process.
8. Only then run the complete 400-example, three-epoch job with final-checkpoint-only saving.

Unknowns that must be resolved by the smoke test rather than guessed:

- whether the chosen Transformers/TRL version supports the Qwen chat template with assistant-only masking;
- whether `all-linear` expands to the intended Qwen projections in the pinned PEFT version;
- whether 16-bit LoRA fits at the chosen max length, or QLoRA is required;
- whether the installed bitsandbytes/CUDA build works correctly on the 3090;
- the largest safe micro-batch and effective sequence length;
- whether FlashAttention/SDPA is available and stable in the selected environment;
- whether packing changes turn boundaries or masks;
- whether the 400-example dataset overfits during three epochs (the issue explicitly excludes checkpoint selection, so log the evidence rather than selecting around it).

## Primary sources

- Qwen model card: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- Qwen source/docs: <https://github.com/QwenLM/Qwen2.5>
- Transformers chat templates: <https://huggingface.co/docs/transformers/main/en/chat_templating>
- TRL SFTTrainer and dataset/loss/config behavior: <https://huggingface.co/docs/trl/en/sft_trainer>
- PEFT LoRA configuration: <https://huggingface.co/docs/peft/en/developer_guides/lora>
- PEFT source: <https://github.com/huggingface/peft>
- Transformers bitsandbytes integration: <https://huggingface.co/docs/transformers/en/quantization/bitsandbytes>
- bitsandbytes integration docs: <https://huggingface.co/docs/bitsandbytes/en/integrations>
- torchtune documentation: <https://docs.pytorch.org/torchtune/stable/>
- torchtune Qwen2.5 7B single-device LoRA recipe: <https://github.com/pytorch/torchtune/blob/main/recipes/configs/qwen2_5/7B_lora_single_device.yaml>
- Unsloth fine-tuning guide: <https://docs.unsloth.ai/get-started/fine-tuning-llms-guide>
- Unsloth source: <https://github.com/unslothai/unsloth>
- Axolotl quickstart: <https://docs.axolotl.ai/docs/getting-started.html>
- Axolotl configuration reference: <https://docs.axolotl.ai/docs/config-reference.html>
- Axolotl source: <https://github.com/axolotl-ai-cloud/axolotl>
