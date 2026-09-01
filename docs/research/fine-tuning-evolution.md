# Fine-tuning evolution: V1 → V2 → V3 → next

This document records how the Socratic tutor fine-tuning experiments changed over time, what each experiment established, and what the current training run is intended to test. The raw pools, notebooks, and run outputs named below are historical identifiers and are intentionally archived from the current release tree; the published evidence under `results/` is the release source of truth.

## Executive summary

The project has kept the base model and core QLoRA recipe stable while changing the training data and, now, the training/evaluation contract:

- **V1** established the Gemma 4 31B QLoRA baseline.
- **V2** added hard-negative SFT examples and improved actionable diagnosis, but did not reduce leakage.
- **V3** added explicit alignment behavior classes, but still leaked and regressed diagnosis utility.
- **Next** fixes the discovered distribution gaps: the exact evaluation system prompt, benchmark-concept coverage with disjoint wording, deeper pressure conversations, contract validation, and epoch snapshots.

The next arm is trained and benchmarked below. Epoch 2 is the selected checkpoint because it is the earliest snapshot with zero judged leakage; its actionable diagnosis is reported honestly rather than hidden behind the retired promotion gate. A separate disjoint benign/holdout suite is not yet present in the repository, so broad over-refusal behavior remains an explicit follow-up check.

## What stayed controlled

Unless explicitly noted, V1, V2, V3, and the next arm use the same core setup:

| Control | Value |
|---|---|
| Base model | `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` |
| Base revision | `8e256fc6d63003fc0ca8c91b976e6dcc38433385` |
| Quantization | Pre-quantized 4-bit QLoRA |
| LoRA coverage | `all-linear`; language, attention, and MLP modules |
| LoRA rank / alpha | 16 / 32 |
| Maximum length | 1,024 tokens |
| Micro-batch / accumulation | 1 / 4; effective batch size 4 |
| Learning rate | `2e-4`, cosine schedule |
| Optimizer | AdamW 8-bit |
| Loss | Assistant responses only |
| Packing | Disabled |
| Seed | `20260825` for model training |
| Benchmark | Fixed 48-case benchmark, temperature 0, seed 0 |

This means the measured V1–V3 differences are primarily data/experiment differences, not attention-only LoRA changes. The next arm intentionally changes the number of training epochs to four only so epochs 2, 3, and 4 can be compared; epoch 3 is the primary three-epoch comparison.

## V1: initial Gemma QLoRA baseline

**Purpose:** establish a reproducible Gemma 4 31B SFT baseline on the 24 GiB RTX 3090.

**Data:** `data/train/dialogues.jsonl`, 400 conversations across four balanced families:

- `normal-stuck`
- `answer-demand`
- `persistent-pressure`
- `misconception-edge`

**Training:** `train/gemma4_qlora_guided.ipynb`, with response-only loss and the original synthetic tutor pool.

**Result:**

- Leakage: **2/48 (4.17%)**
- Actionable diagnosis: **32/48 (66.67%)**

V1 demonstrated that QLoRA could improve the prompted base model's leakage score from 4/48 to 2/48, but it did not establish a zero-leakage boundary.

Evidence:

- `train/config.yaml`
- `runs/gemma4_adapter/eval_results.jsonl`
- `data/train/README.md`

## V2: hard-negative SFT pool

**Purpose:** make answer-demand and pressure behavior more explicit while keeping the model and optimizer recipe fixed.

**Data:** `data/train-v2/dialogues.jsonl`, a separate 400-record pool with hard-negative replacements. The pool preserved the four 100-record family balance.

**Training:** `train/gemma4_qlora_v2.ipynb`.

**Result:**

- Leakage: **3/48 (6.25%)**
- Actionable diagnosis: **43/48 (89.58%)**

V2 improved diagnosis utility by 11 cases but did not improve leakage. Its failures showed that generic refusal-plus-diagnosis SFT can teach the style without reliably suppressing semantic leaks such as a built-in name, an initialization strategy, or exact boundary logic.

The one-case leakage difference from V1 is only 2.08 percentage points on a 48-case benchmark, so it should be treated as an ablation signal rather than proof that V2 is generally worse.

Evidence:

- `train/config-v2.yaml`
- `runs/gemma4_v2_adapter/eval_results.jsonl`
- `data/train-v2/README.md`

## V3: alignment-class SFT pool

**Purpose:** target known artifact classes directly instead of balancing only by conversation family.

**Data:** `data/train-v3/dialogues.jsonl`, still 400 records, with 80 replacement records distributed across five behavior classes:

1. numeric/final-result demand;
2. formula or condition demand;
3. built-in/API demand;
4. algorithm or initialization confirmation;
5. instruction-hierarchy conflict.

The pool also included `data/train-v3/behavior_balance.csv` and `semantic_review.md`.

**Training:** `train/gemma4_qlora_v3.ipynb`, using the same rank, coverage, optimizer, three-epoch schedule, and assistant-only SFT objective.

**Result:**

- Leakage: **3/48 (6.25%)**
- Actionable diagnosis: **30/48 (62.50%)**

V3 did not improve leakage and fell below V1 on diagnosis utility. The recorded failures were:

- exact numeric output (`24`);
- a complete leap-year rule in prose;
- the `max()` built-in name.

The result exposed four practical gaps:

- training used a generic system prompt rather than the evaluator's canonical prompt;
- training exercises were disjoint from the benchmark concepts, including maximums, leap years, rectangles, and grade boundaries;
- persistent-pressure training contained only one escalation;
- SFT still rewarded chosen responses but never explicitly penalized leaking alternatives.

Evidence:

- `train/config-v3-sft.yaml`
- `runs/gemma4_v3_sft_adapter/eval_results.jsonl`
- `data/train-v3/README.md`
- the archived V3 alignment plan

## Next: clean SFT plus epoch comparison

**Purpose:** fix the observed distribution and measurement problems before testing preference optimization or larger adapters.

**Status:** complete. Smoke validation passed, the four-epoch run completed, and all three saved snapshots were evaluated against the fixed benchmark.

### Data changes

`data/train-next/` is generated with the alignment profile in `scripts/generate_dialogues.py`:

- exact `eval.judge.TUTOR_SYSTEM_PROMPT` in every row;
- 400 records with the same four balanced families;
- disjoint exercises covering:
  - maximum-style list scans;
  - layered calendar/divisibility rules;
  - rectangle measurements;
  - numeric score boundaries;
- concept-specific diagnosis language;
- persistent-pressure conversations expanded to seven messages;
- deterministic target-phrasing variation;
- benchmark-overlap checking and manifest generation.

The archived next pool was validated with this historical command (the pool is
not included in the release):

```bash
python3 scripts/validate_train.py data/train-next \
  --require-canonical-system-prompt
```

The shared contract implementation is `scripts/training_contract.py`. It checks the canonical system prompt, alternating role structure, non-empty messages, and the exact rows hash that enters training.

### Historical training changes

The archived `train/gemma4_qlora_next.ipynb` preserved the V3 model and optimizer
settings but:

- consumed `data/train-next/`;
- recorded the canonical prompt hash and training-row hash;
- proved the assistant-only response mask;
- saved adapter snapshots at epochs 2, 3, and 4;
- wrote the full-run configuration and canonicalized training rows;
- refused to overwrite an existing next-run adapter directory.

Historical full-run artifacts (not shipped):

```text
train/adapter-next-sft/epoch-2/
train/adapter-next-sft/epoch-3/
train/adapter-next-sft/epoch-4/
train/config-next-sft.yaml
train/logs-next-sft/
train/adapter-next-sft.sha256
```

### Why four epochs?

Four is not a claim that more training is better. It is the smallest run that supplies the requested 2/3/4 epoch comparison in one controlled training job. The benchmark selected epoch 2 as the earliest zero-leakage checkpoint; the 32/48 diagnosis result is reported honestly rather than used to manufacture a pass gate. Epoch 4 was not preferred automatically.

### Measured result

All three evaluations used the fixed 48-case benchmark, canonical tutor prompt, base revision above, `judge-v1`, temperature `0`, seed `0`, and greedy local decoding with a 256-token cap. Raw transcripts, judge calls, reports, and evaluation configuration are under `runs/gemma4_next_epoch-2/`, `runs/gemma4_next_epoch-3/`, and `runs/gemma4_next_epoch-4/`.

| Snapshot | Leakage | Actionable diagnosis | Normal-stuck | Answer-demand | Persistent-pressure | Misconception-edge | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Epoch 2 | **0/48 (0.00%)** | **32/48 (66.67%)** | 12/12 | 6/12 | 3/12 | 11/12 | **Selected: earliest zero-leakage checkpoint** |
| Epoch 3 | **0/48 (0.00%)** | **33/48 (68.75%)** | 12/12 | 5/12 | 4/12 | 12/12 | Later; +1 diagnosis vs. selected |
| Epoch 4 | **0/48 (0.00%)** | **33/48 (68.75%)** | 12/12 | 5/12 | 5/12 | 11/12 | Later; no overall gain over epoch 3 |

The ordinary `normal-stuck` family remained leak-free with 12/12 actionable diagnoses for every snapshot. No visible chat-template or reasoning markers were found in assistant outputs. The repository has no separate disjoint benign/holdout pool, so this family result is useful evidence but does not replace that missing suite. The conservative lexical smoke scorer flagged code-like tutoring language; the pinned semantic judge scored zero leakage and remains authoritative for the experiment.

## Historical run sequence

The archived run:

1. ran notebook inspection and smoke gates;
2. confirmed hardware, dependency, prompt, rendered-template, response-mask,
   loss, and memory conditions;
3. ran full mode only after smoke succeeded;
4. evaluated epochs 2, 3, and 4 against the same fixed 48-case benchmark;
5. reported leakage and diagnosis as counts and rates with raw supporting records;
6. selected the earliest zero-leakage checkpoint rather than selecting by training
   loss alone.

The fixture-first release does not execute this training workflow. The selected
checkpoint path is retained in published evidence as a historical identifier.

## Acceptance and stop conditions

Promote a next checkpoint only if all of the following hold:

- leakage is no worse than V1: **≤2/48**;
- actionable diagnosis is at least V1: **≥32/48**;
- the available ordinary-tutoring benchmark family shows no regression; a separate benign/holdout suite is still required for a complete broad promotion decision;
- no visible `thought`/template fragments remain in user-facing output;
- prompt/template hashes and adapter/data/config manifests are saved.

Stop and revise the data or serving contract if:

- leakage increases;
- diagnosis is achieved mainly through repeated refusal boilerplate;
- exact built-ins, formulas, values, or algorithms continue to leak;
- epoch 4 adds repetition without improving the boundary;
- train/inference formatting differs;
- the missing benign/holdout suite is added and reveals material over-refusal.

## What is deliberately not changed yet

The next arm does **not** yet test:

- DPO, ORPO, or SimPO;
- rank 32 or rank 64;
- full fine-tuning;
- packing;
- multiple training seeds;
- adaptive prompt-injection resistance;
- a runtime output filter as a replacement for training.

If clean SFT still plateaus, the next controlled branch is ORPO with new preference pairs: chosen responses should refuse the artifact while diagnosing the issue; rejected responses should contain a semantic leak or a sterile refusal. Keep those pairs separate from the fixed benchmark.
