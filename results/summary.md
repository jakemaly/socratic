# Fixed benchmark evidence

## Selected result

The selected checkpoint is **`train/adapter-next-sft/epoch-2/`**. On the fixed
48-case benchmark, it scored **0/48 judged leakage** and **32/48 actionable
diagnosis**. The selection is the earliest checkpoint in this run with zero
judged leakage; it is not a claim that the tutor is universally leak-proof.

## Comparison

| Model | Checkpoint | Judged leakage | Actionable diagnosis | Decision |
| --- | --- | ---: | ---: | --- |
| Gemma 4 base | prompted comparison snapshot | 4/48 (8.33%) | 35/48 (72.92%) | comparison |
| Gemma 4 + LoRA | epoch 2 | 0/48 (0.00%) | 32/48 (66.67%) | **selected** |
| Gemma 4 + LoRA | epoch 3 | 0/48 (0.00%) | 33/48 (68.75%) | later checkpoint |
| Gemma 4 + LoRA | epoch 4 | 0/48 (0.00%) | 33/48 (68.75%) | later checkpoint |

Epoch 2 reduces judged leakage by 4/48
(8.33 percentage points) versus the
base snapshot. Its actionable-diagnosis result is -3/48
(-6.25 percentage points) versus base.
No promotion thresholds are applied to this presentation result.

## Verdict

The evidence supports using epoch 2 for the portfolio project because it removes
all judged leakage on this fixed benchmark. It does **not** support a universal
leak-proof claim: diagnosis is 32/48, calibration was not performed, and no
disjoint benign/holdout suite was run.

## Family breakdown

| Family | Base leakage | Base diagnosis | Epoch 2 leakage | Epoch 2 diagnosis | Epoch 3 diagnosis | Epoch 4 diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `normal-stuck` | 1/12 (8.33%) | 12/12 (100.00%) | 0/12 (0.00%) | 12/12 (100.00%) | 12/12 (100.00%) | 12/12 (100.00%) |
| `answer-demand` | 1/12 (8.33%) | 5/12 (41.67%) | 0/12 (0.00%) | 6/12 (50.00%) | 5/12 (41.67%) | 5/12 (41.67%) |
| `persistent-pressure` | 2/12 (16.67%) | 6/12 (50.00%) | 0/12 (0.00%) | 3/12 (25.00%) | 4/12 (33.33%) | 5/12 (41.67%) |
| `misconception-edge` | 0/12 (0.00%) | 12/12 (100.00%) | 0/12 (0.00%) | 11/12 (91.67%) | 12/12 (100.00%) | 11/12 (91.67%) |


## Methodology

- Benchmark: `data/benchmark/cases.jsonl` (SHA-256 `78a0d1e3d96e6b108c73f51469b84ebb7b2ae02e0c586aaea32559fb938aad4e`)
- Judge: `deepseek-v4-pro`, `judge-v1`
- Temperature: `0`; seed: `0`; decoding: greedy; maximum new tokens: `256`
- Base model and all checkpoints use the same fixed benchmark and canonical tutor prompt.
- Raw evaluation files were produced under `runs/gemma4_base/` and
  `runs/gemma4_next_epoch-2/` through `runs/gemma4_next_epoch-4/`.

## Limitations

- The claim is **0/48 judged leakage on the fixed benchmark**, not universal leak-proof behavior.
- There is no separate human calibration result, so calibration is **not performed** in the presentation record.
- There is no disjoint benign/holdout suite; broad over-refusal behavior remains follow-up research.
- The judge's semantic labels are evidence for this benchmark, not a substitute for broader human evaluation.
