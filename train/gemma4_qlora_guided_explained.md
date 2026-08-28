# Gemma 4 QLoRA Training Notebook: a beginner-to-complete guide

This document explains **`train/gemma4_qlora_guided.ipynb`**, the canonical
training notebook in this repository. It is written for someone who knows a
little Python but is new to machine learning.

It explains two things at once:

1. **The big picture:** what the experiment is trying to teach the model, what
   QLoRA and supervised fine-tuning mean, what happens to the data, and what the
   smoke run does and does not prove.
2. **The notebook itself:** every executable notebook line, in order, including
   lines that only print information or enforce a safety check.

The notebook currently defaults to `RUN_MODE = 'smoke'`. That is deliberate.
Do not interpret a passing smoke run as a trained model or as evidence that the
model improved. It proves that one real training update can run with this data,
model, precision, and GPU configuration.

> **Canonical notebook warning:** `train/qwen_lora_guided.ipynb` is an older,
> historical 7B BF16 LoRA experiment. The notebook explained here is the
> canonical recipe: Gemma 4 31B, Unsloth, 4-bit loading, and QLoRA.

---

## 1. The one-sentence explanation

You are taking a pretrained Gemma 4 31B instruction-following language model,
showing it 400 carefully filtered examples of how a Socratic Python tutor should
respond, and learning a small set of additional adapter weights—without
rewriting the original model—using three passes over the examples.

The adapter is trained with **supervised fine-tuning (SFT)**. For every example,
the model sees a conversation and is trained to predict the next token in the
assistant's response. The loss is calculated on response text, not on the
system/user instructions. The base model is loaded in 4-bit form to fit on the
24 GiB RTX 3090; the trainable LoRA matrices remain the small learned change.

The desired behavioral change is not “make Gemma know Python.” Gemma already
knows a great deal of Python. The desired change is closer to:

- ask guiding questions;
- diagnose the learner's likely conceptual mistake;
- provide useful next steps and resources; and
- refuse to provide completed code or final answers.

The notebook trains the adapter. It does **not** decide whether the behavior is
good. That judgment happens afterward in the separate fixed 48-case benchmark.

---

## 2. What happens when you run it

The notebook is a sequence of gates. A gate is a check that stops the run rather
than allowing a questionable experiment to continue.

### In `inspect` mode

- Python and package/hardware information is inspected.
- The 400 training records are checked.
- The dataset is converted and shuffled.
- The model is **not** loaded.
- No GPU generation happens.
- No training happens.
- No adapter is saved.

This is the cheapest way to read through the data side of the notebook.

### In `smoke` mode (the default)

- The same fixed model and data path are used as full training.
- The exact pinned 4-bit Gemma model is loaded.
- Three small base-model replies are generated as a preflight.
- The conversations are formatted with Gemma's chat template.
- LoRA adapters are attached.
- The response-only loss wrapper is installed and checked.
- **Exactly one optimizer step** is requested with `max_steps=1`.
- The temporary adapter is saved to a temporary directory.
- The temporary result is not a final experiment artifact.

### In `full` mode

- It performs the same setup and gates.
- It runs for three epochs over the 400 records.
- It writes the adapter and evidence under `train/`.
- It reloads the saved adapter into a fresh base model.
- It generates one independent sanity-check reply.

Full mode requires changing **both**:

```python
RUN_MODE = 'full'
CONFIRM_FULL_RUN = True
```

That second setting prevents accidentally starting a long GPU job merely by
changing one string.

---

## 3. The exact experiment, without hand-waving

| Item | Notebook choice | What it means |
|---|---|---|
| Base model | `unsloth/gemma-4-31B-it-unsloth-bnb-4bit` | Gemma 4 31B instruction-tuned model, in Unsloth's pre-quantized 4-bit artifact |
| Model identity | Hub revision `8e256fc6d63003fc0ca8c91b976e6dcc38433385` | A fixed model snapshot rather than a moving model name |
| Model loading | `FastModel.from_pretrained(..., load_in_4bit=True)` | Load the frozen base in 4-bit to reduce VRAM |
| Training method | supervised fine-tuning | Learn from target assistant responses with next-token prediction |
| Parameter method | QLoRA | Quantized base plus trainable low-rank LoRA adapters |
| Vision training | off | This dataset is text-only |
| Language training | on | The language model is the part being adapted |
| Attention modules | on | LoRA can change attention projections |
| MLP modules | on | LoRA can change feed-forward projections |
| LoRA rank | 16 | Size of each low-rank update; larger means more capacity and more memory |
| LoRA alpha | 32 | Scaling factor for the adapter update |
| LoRA dropout | 0 | No random adapter dropout |
| RS-LoRA | off | Standard LoRA scaling, not rank-stabilized LoRA |
| Data | 400 committed training dialogues | The notebook does not regenerate them |
| Data structure | `turns` renamed to `messages` | Standard role/content chat records |
| Ordering | Shuffled with seed `20260825` | Same content, reproducible order in normal conditions |
| Context cap | 1,024 tokens | Longer examples are rejected by the notebook's formatting check |
| Micro-batch | 1 example | One example is processed at a time on the GPU |
| Gradient accumulation | 4 | Four micro-batches contribute to one optimizer update |
| Nominal effective batch | 4 examples | `1 × 4`; this is not four examples simultaneously in VRAM |
| Epochs | 3 in full mode | The 400-record pool is read three times |
| Learning rate | `2e-4` | Initial size of optimizer updates |
| Warmup | 5 optimizer steps | The learning rate ramps up at the beginning |
| Schedule | cosine | Learning rate follows a cosine decay after warmup |
| Optimizer | `adamw_8bit` | AdamW state stored in an 8-bit memory-saving form |
| Weight decay | `0.001` | Small regularizing pull on weights |
| Gradient clipping | max norm `0.3` | Prevent unusually large updates |
| Math | BF16, with TF32 enabled | Fast reduced-precision GPU computation appropriate to the RTX 3090 |
| Memory saving | Unsloth gradient checkpointing | Recompute some activations instead of storing all of them |
| Loss | response-only | Assistant/model spans are targets; instructions are masked out |
| Packing | off | Do not concatenate separate conversations into one sequence |
| Evaluation during training | none | There is no validation set or checkpoint selection |
| Saving | final adapter only | No collection of intermediate checkpoints |
| Seed | `20260825` | Passed to Python, PyTorch, CUDA, Unsloth, and trainer configuration |

The approximate full-run arithmetic is:

- 400 records per epoch;
- 3 epochs = 1,200 training examples seen;
- micro-batch size 1 = about 1,200 micro-batches;
- accumulation of 4 = about 300 optimizer updates.

The exact last-step behavior can depend on trainer batching details, so treat
300 as the nominal number, not as a separately recorded result.

---

## 4. ML concepts from zero

### 4.1 A language model is a next-token guesser

A language model does not start with a database of complete answers. It reads a
sequence of tokens and assigns probabilities to possible next tokens.

For example, a tokenizer might split:

```text
What should I check next?
```

into token pieces such as:

```text
What | should | I | check | next | ?
```

The actual pieces and IDs are model-specific. A **token ID** is just an integer
used to look up a token in the model's vocabulary.

At generation time, the model repeatedly does this:

1. read the tokens so far;
2. calculate a score, called a **logit**, for every possible next token;
3. turn scores into probabilities;
4. choose a token according to the decoding settings;
5. append it and repeat.

`tokenizer` converts human-readable text to token IDs and converts generated IDs
back to text. `model` maps token IDs to next-token scores.

### 4.2 Weights are the model's learned numbers

A neural network is mostly a very large collection of numeric tensors called
**weights** or **parameters**. Pretraining adjusted those numbers so the model
could predict text across a huge corpus. The notebook does not retrain all of
those numbers.

A parameter with `requires_grad=True` is allowed to receive a gradient and be
updated by the optimizer. A frozen parameter is used for predictions but is not
changed by this run.

### 4.3 Forward pass, loss, backward pass, optimizer step

One training example goes through this loop:

1. **Forward pass:** the model reads token IDs and produces next-token scores.
2. **Loss:** the scores are compared with the known next tokens. Cross-entropy
   loss is high when the model assigns low probability to the expected token and
   low when it assigns high probability.
3. **Backward pass:** automatic differentiation calculates how each trainable
   parameter contributed to the loss. These numbers are gradients.
4. **Gradient accumulation:** gradients from four micro-batches are collected.
5. **Optimizer step:** AdamW uses the gradients to modify the LoRA parameters.
6. **Gradient reset:** the next accumulation cycle starts cleanly.

The model is trained with **teacher forcing**: while learning a response, it is
shown the correct previous tokens rather than having to generate its own entire
response first. This makes training efficient, but it is different from
real-world generation, which is why the separate benchmark matters.

### 4.4 Why only response tokens count

A formatted example contains instructions and answers. For example:

```text
<|turn>system
... tutor rules ...<turn|>
<|turn>user
I am stuck ...<turn|>
<|turn>model
Ask me what inputs I have identified.<turn|>
```

The user text is context. The model should read it in order to produce a good
answer, but the experiment does not want to reward it for copying the system or
user text. The training labels therefore use `-100` for ignored tokens and real
token IDs for assistant spans.

In the notebook, `train_on_responses_only` recognizes:

```text
instruction_part = '<|turn>user\n'
response_part    = '<|turn>model\n'
```

The response-only gate checks that both kinds of labels exist:

- some labels are `-100`, meaning instruction tokens are ignored;
- some labels are not `-100`, meaning there are assistant tokens to learn.

This is a training objective choice, not a claim that the model cannot read the
instructions. It absolutely reads them as input context.

### 4.5 LoRA: learn a small change instead of rewriting the house

Suppose a normal linear layer uses a large weight matrix `W`:

```text
output = x Wᵀ
```

LoRA keeps `W` frozen and adds a learned update represented as two small
matrices, often written `A` and `B`:

```text
output = x Wᵀ + (x Aᵀ Bᵀ) × (alpha / rank)
```

The product `AᵀBᵀ` behaves like an update to `W`, but it is represented with
many fewer trainable values when `rank` is small.

ELI5: the pretrained model is a house. Full fine-tuning repaints every wall.
LoRA leaves the house intact and learns a small set of removable stickers that
change how it behaves. The adapter directory contains the stickers, not a
second complete 31B model.

With rank 16 and alpha 32, the notebook gives the adapters enough capacity to
change behavior while keeping the trainable portion much smaller than the
frozen base. `bias='none'` means biases are not trained as an additional path.
`use_rslora=False` selects ordinary LoRA scaling.

### 4.6 QLoRA: LoRA with a quantized base

**Quantization** stores numbers with fewer bits. A 4-bit representation uses far
less memory than BF16 or FP16, at the cost of approximation. QLoRA combines:

- a base model loaded in 4-bit;
- trainable LoRA adapters; and
- higher-precision computation where needed for training operations.

The base is still useful because it contains the pretrained knowledge. The
training updates are focused on the small adapters. This is why a 31B model can
be attempted on a 24 GiB card at all.

This notebook is not “full fine-tuning in 4-bit.” It is not changing every base
weight. It is also not the same experiment as ordinary BF16 LoRA: quantization
mode is part of the method and is recorded as QLoRA in the full-run config.

### 4.7 Attention and MLP modules

A transformer layer has several broad jobs:

- **Attention** lets each token use information from other tokens. Its query,
  key, value, and output projections decide how information is selected and
  mixed.
- **MLP/feed-forward blocks** transform each token's representation through
  learned nonlinear layers.

The Unsloth call enables language-layer adapters, attention-module adapters, and
MLP-module adapters, while disabling vision-layer adapters. That is appropriate
for the text-only tutor data. It does not mean a separate vision model is being
trained.

### 4.8 Precision and memory settings

- **BF16** is a 16-bit floating-point format with a wide exponent range. It is
  used for the computation supported by the chosen GPU recipe.
- **TF32** is an NVIDIA acceleration mode for some matrix operations. It trades
  mantissa precision for speed in ways intended to be acceptable for this use.
- **Gradient checkpointing** stores fewer intermediate activations during the
  forward pass and recomputes them during backpropagation. It saves VRAM but
  costs extra compute time.
- **Unsloth gradient checkpointing** is the specifically selected memory-saving
  implementation for this 31B recipe.
- **Micro-batch size 1** means only one example is held in the per-device batch.
- **Gradient accumulation 4** makes the optimizer behave approximately as if it
  had a batch of four, while using the memory of one example at a time.

The 1,024-token limit is important because activation memory generally grows as
sequences get longer. The notebook rejects a formatted training record that is
longer than this cap instead of silently training on truncated content.

### 4.9 The optimizer and schedule

AdamW is an optimizer: a rule for turning gradients into parameter updates. The
notebook uses the bitsandbytes-backed `adamw_8bit` variant so optimizer state
uses less memory.

The learning rate `2e-4` controls update size. It is not the loss and it is not a
probability. A learning rate that is too large can make training unstable; one
that is too small can make learning painfully slow.

The schedule is:

1. start with a short five-step warmup;
2. then reduce the learning rate according to a cosine curve.

`weight_decay=0.001` is a small regularization setting. `max_grad_norm=0.3`
clips unusually large gradients before the optimizer uses them. These are fixed
choices, not values selected by an automated search.

---

## 5. The data and the purpose of the examples

The notebook reads `data/train/dialogues.jsonl`. The repository validator
requires 400 records, and each record has this role pattern:

```text
system → user → assistant → user → assistant
```

The four data families are intended to cover different pressure situations:

- `normal-stuck`: a learner is stuck and needs a useful next question;
- `answer-demand`: a learner asks for the completed answer;
- `persistent-pressure`: the learner continues pressing after redirection;
- `misconception-edge`: the learner has a plausible but wrong assumption.

The training pool is not regenerated by the notebook. It is treated as a
committed input. The validator checks schema and gates, while the SHA-256
manifest checks that the file's bytes match the committed expected fingerprint.

The training records include system messages that describe the tutor behavior.
The preflight fixtures use the canonical `TUTOR_SYSTEM_PROMPT` imported from
`eval/judge.py`, so the preflight is asking the base model under the same prompt
contract used by evaluation. The notebook does not score those three replies;
it prints them for a human to inspect.

### One training record, end to end

For one row, the practical data path is:

```text
JSONL row
  → row['turns'] renamed to a `messages` list
  → Hugging Face Dataset row (`id`, `messages`)
  → deterministic shuffle
  → Gemma chat template
  → one `text` string with Gemma special turn markers
  → token IDs, limited to 1,024 positions
  → labels with instruction positions set to -100
  → model forward pass
  → assistant-token cross-entropy loss
  → gradients on LoRA tensors only
  → four micro-batches accumulated
  → one AdamW 8-bit update
```

The model input and label arrays have the same token positions. `-100` is a
special PyTorch loss value meaning “ignore this position.” It is not a token
that the model learns to output. A typical batch has shape conceptually like
`[batch, sequence_length]`; here the batch dimension is 1 and sequence length
is at most 1,024. The exact number of tokens varies by conversation.

The `gemma-4-thinking` name deserves a precise note. It selects Gemma's
thinking-capable chat formatting, but this notebook does not invent a separate
chain-of-thought dataset or manually add hidden reasoning text. The actual
rendered string printed by Cell 13 is the source of truth. Every assistant text
that remains unmasked is a training target, regardless of what the template is
called. If the rendered format or any thought markers look surprising, stop and
inspect them before full training.

The separate benchmark is deliberately not run inside training. It should use
the same 48 cases, model revision, tokenizer/chat behavior, decoding settings,
and judge for the base and adapter arms. The benchmark asks whether the adapter
changed the two important outcomes:

- **leakage:** did the tutor provide completed code/final output?
- **actionable diagnosis:** did it identify the likely conceptual error and ask
a relevant next-step question?

A lower training loss alone cannot prove either behavioral outcome.

---

## 6. Cell-by-cell and line-by-line explanation

Notebook Markdown cells are explained as sections here. Every executable line is
covered below. Blank lines in code cells are visual separators and have no
runtime effect; they are still noted where useful. A line containing several
Python expressions is explained piece by piece rather than treated as magic.

### Cells 1–3: the notebook's own introduction

These Markdown cells state the model, GPU constraint, three modes, and fixed
experiment. They do not execute Python and cannot change weights. Their most
important practical instruction is “run one cell at a time”: the notebook is
built as a lesson and as a sequence of gates, not as an opaque Run All script.

### Cell 4: imports, constants, and mode lock

```python
 1 from __future__ import annotations
 2
 3 import hashlib
 4 import importlib.metadata
 5 import json
 6 import os
 7 import random
 8 import sys
 9 import subprocess
10 import tempfile
11 import time
12 from collections import Counter
13 from pathlib import Path
14
15 import torch
16
17 def find_repo_root(start: Path) -> Path:
18     for candidate in (start, *start.parents):
19         if (candidate / 'data/train/dialogues.jsonl').exists():
20             return candidate
21     raise FileNotFoundError('Open this notebook from inside the socratic repository.')
22
23 ROOT = find_repo_root(Path.cwd())
24 if str(ROOT) not in sys.path:
25     sys.path.insert(0, str(ROOT))
26 MODEL_ID = 'unsloth/gemma-4-31B-it-unsloth-bnb-4bit'
27 MODEL_REVISION = '8e256fc6d63003fc0ca8c91b976e6dcc38433385'
28 MAX_SEQ_LENGTH = 1024
29 LORA_R = 16
30 LORA_ALPHA = 32
31 LORA_DROPOUT = 0.0
32 LEARNING_RATE = 2e-4
33 EPOCHS = 3
34 MICRO_BATCH_SIZE = 1
35 GRADIENT_ACCUMULATION_STEPS = 4
36 SEED = 20260825
37 RUN_MODE = 'smoke'
38 CONFIRM_FULL_RUN = False
39 DATA_PATH = ROOT / 'data/train/dialogues.jsonl'
40 MANIFEST_PATH = ROOT / 'data/train/manifest.sha256'
41 FIXTURE_PATH = ROOT / 'data/fixtures/benchmark_cases.jsonl'
42 FINAL_ADAPTER_DIR = ROOT / 'train/adapter'
43 FINAL_LOG_DIR = ROOT / 'train/logs'
44
45 if RUN_MODE not in {'inspect', 'smoke', 'full'}:
46     raise ValueError('RUN_MODE must be inspect, smoke, or full')
47 if RUN_MODE == 'full' and not CONFIRM_FULL_RUN:
48     raise RuntimeError('Set CONFIRM_FULL_RUN=True only after reading smoke output.')
49
50 random.seed(SEED)
51 torch.manual_seed(SEED)
52 if torch.cuda.is_available():
53     torch.cuda.manual_seed_all(SEED)
54 os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
55 print('Repository:', ROOT)
56 print('Model:', MODEL_ID)
57 print('Revision:', MODEL_REVISION)
58 print('Mode:', RUN_MODE)
```

Line by line:

- **1:** Enables postponed evaluation of type annotations. It makes annotations
  such as `Path` and modern union types safer to evaluate consistently.
- **2:** Blank separator.
- **3:** Imports SHA-256 hashing, used for data and adapter fingerprints.
- **4:** Imports Python's installed-package metadata reader.
- **5:** Imports JSON parsing and serialization.
- **6:** Imports operating-system helpers, including environment variables.
- **7:** Imports Python's pseudorandom-number generator.
- **8:** Imports interpreter and import-path information.
- **9:** Imports subprocess execution for the repository validator.
- **10:** Imports temporary-directory support for smoke output.
- **11:** Imports a wall-clock timer for training duration.
- **12:** Imports `Counter`, used to count training families.
- **13:** Imports `Path`, Python's safer path abstraction.
- **14:** Blank separator.
- **15:** Imports PyTorch, which supplies tensors, GPU access, automatic
  differentiation, and the low-level training machinery.
- **16:** Blank separator.
- **17:** Defines a function named `find_repo_root`; `start: Path` is its input
  annotation and `-> Path` says it returns a path.
- **18:** Iterates over the current candidate and every parent directory. The
  `*` expands the tuple of parent paths into the surrounding tuple.
- **19:** Tests for the repository's committed training file. This is the marker
  that identifies the project root.
- **20:** Returns the first directory containing that marker.
- **21:** If no candidate worked, stops with a useful error instead of reading a
  similarly named file from the wrong checkout.
- **22:** Blank separator.
- **23:** Gets Jupyter's current working directory with `Path.cwd()` and finds
  the repository root from it.
- **24:** Checks whether the root is already on Python's module search path.
- **25:** Adds the root at the front of `sys.path` when needed, so imports such
  as `eval.judge` resolve to this checkout.
- **26:** Names the exact pretrained model artifact to load.
- **27:** Pins the Hub revision. The model name is a label; this hash identifies
  one immutable version of its files.
- **28:** Caps each training sequence at 1,024 tokens.
- **29:** Sets LoRA rank to 16.
- **30:** Sets the LoRA scaling value to 32.
- **31:** Sets adapter dropout to zero, meaning no random adapter units are
  dropped during training.
- **32:** Sets the learning rate to 0.0002.
- **33:** Requests three complete passes through the data in full mode.
- **34:** Sets one example per device micro-batch.
- **35:** Accumulates gradients over four micro-batches before an optimizer step.
- **36:** Declares the run's reproducibility seed.
- **37:** Selects the safe default, one-step smoke mode.
- **38:** Leaves the second full-run confirmation disabled.
- **39:** Points at the committed 400-record JSONL training file.
- **40:** Points at the file containing the expected SHA-256 digest.
- **41:** Points at fixture cases used only for the small preflight.
- **42:** Names where the final adapter will be written in full mode.
- **43:** Names where full-run logs will be written.
- **44:** Blank separator.
- **45:** Checks that the mode is one of the three supported strings.
- **46:** Raises a clear error for a typo or unsupported mode.
- **47:** Detects the dangerous combination of full mode without explicit
  confirmation.
- **48:** Stops that combination before model loading or GPU work.
- **49:** Blank separator.
- **50:** Seeds Python's random generator.
- **51:** Seeds PyTorch's random generator.
- **52:** Checks whether CUDA is available before trying CUDA-specific seeding.
- **53:** Seeds all visible CUDA devices if they exist.
- **54:** Disables tokenizer worker parallelism unless the environment already
  set a value. This reduces noisy worker behavior and some reproducibility
  surprises.
- **55:** Prints the resolved repository root as a receipt.
- **56:** Prints the selected model ID.
- **57:** Prints the selected immutable revision.
- **58:** Prints the selected mode.

### Cell 5: package, RAM, GPU, and dependency gate

```python
 1 def package_version(name: str) -> str:
 2     try:
 3         return importlib.metadata.version(name)
 4     except importlib.metadata.PackageNotFoundError:
 5         return 'missing'
 6
 7 def memory_gib() -> tuple[float | None, float | None]:
 8     values = {}
 9     try:
10         for line in Path('/proc/meminfo').read_text().splitlines():
11             key, value, *_ = line.split()
12             if key in {'MemTotal:', 'MemAvailable:'}:
13                 values[key] = int(value) / 1024 / 1024
14     except (FileNotFoundError, ValueError):
15         return None, None
16     return values.get('MemTotal:'), values.get('MemAvailable:')
17
18 total_ram, available_ram = memory_gib()
19 gpu_available = torch.cuda.is_available()
20 gpu_name = torch.cuda.get_device_name(0) if gpu_available else 'CPU only'
21 gpu_vram = (torch.cuda.get_device_properties(0).total_memory / 2**30) if gpu_available else None
22 bf16_supported = gpu_available and torch.cuda.is_bf16_supported()
23 print('Python:', os.sys.version.split()[0])
24 print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
25 print('GPU:', gpu_name)
26 print('GPU VRAM GiB:', f'{gpu_vram:.1f}' if gpu_vram else 'n/a')
27 print('BF16 supported:', bf16_supported)
28 print('System RAM GiB:', f'{total_ram:.1f}' if total_ram else 'unknown')
29 print('Available RAM GiB:', f'{available_ram:.1f}' if available_ram else 'unknown')
30 for package in ('unsloth', 'unsloth-zoo', 'bitsandbytes', 'transformers', 'trl', 'datasets', 'accelerate'):
31     print(f'{package}: {package_version(package)}')
32
33 if RUN_MODE != 'inspect':
34     if not gpu_available or (gpu_vram is not None and gpu_vram < 22):
35         raise RuntimeError('Gemma 4 31B QLoRA smoke/full mode needs a GPU with about 22 GiB available.')
36     if not bf16_supported:
37         raise RuntimeError('The 3090 recipe requires BF16 support.')
38     if package_version('unsloth') == 'missing' or package_version('bitsandbytes') == 'missing':
39         raise RuntimeError('Install Unsloth and bitsandbytes from train/requirements.txt first.')
40     print('Hardware and Unsloth dependency gate: PASS')
```

- **1:** Defines a helper that accepts a distribution name and returns a text
  version.
- **2:** Starts a block that may fail if the package is not installed.
- **3:** Reads the installed distribution version.
- **4:** Catches the specific “package not installed” exception.
- **5:** Reports `missing` instead of crashing during reporting.
- **6:** Blank separator.
- **7:** Defines a helper returning either two floating-point GiB values or
  `None` values.
- **8:** Creates a dictionary for the memory values found.
- **9:** Starts the Linux `/proc` read attempt.
- **10:** Reads `/proc/meminfo` and splits it into lines.
- **11:** Splits each memory line into its key, value, and any remaining fields.
- **12:** Keeps only total and currently available memory.
- **13:** Converts the Linux KiB value to GiB: dividing by 1,024 twice.
- **14:** Handles a non-Linux machine or malformed memory file.
- **15:** Returns unknown values if the memory probe cannot run.
- **16:** Returns the two selected values, or `None` when absent.
- **17:** Blank separator.
- **18:** Runs the RAM helper and stores total and available RAM.
- **19:** Asks PyTorch whether CUDA is usable.
- **20:** Gets GPU zero's human-readable name, or says CPU-only.
- **21:** Gets GPU zero's total VRAM and converts bytes to GiB.
- **22:** Requires both CUDA and BF16 support for a true result.
- **23:** Prints the Python version.
- **24:** Prints PyTorch and the CUDA version PyTorch was built against.
- **25:** Prints the GPU name.
- **26:** Prints VRAM with one decimal place, or `n/a`.
- **27:** Prints the BF16 capability.
- **28:** Prints total system RAM with one decimal place.
- **29:** Prints currently available RAM.
- **30:** Loops over the important training distributions.
- **31:** Prints each package's installed version via the helper.
- **32:** Blank separator.
- **33:** Applies the hard hardware/dependency checks only in smoke/full modes.
- **34:** Rejects a missing GPU or a card with less than the approximately 22
  GiB target.
- **35:** Raises the hardware error. There is no silent CPU or lower-precision
  fallback.
- **36:** Checks BF16 support.
- **37:** Stops if the fixed recipe cannot use BF16.
- **38:** Checks both Unsloth and bitsandbytes installation status.
- **39:** Stops with the repository installation instruction if either is absent.
- **40:** Prints the gate receipt after all those checks pass.

### Cell 7: hash and validate the training pool

```python
 1 def sha256(path: Path) -> str:
 2     digest = hashlib.sha256()
 3     with path.open('rb') as handle:
 4         for chunk in iter(lambda: handle.read(1024 * 1024), b''):
 5             digest.update(chunk)
 6     return digest.hexdigest()
 7
 8 rows = [json.loads(line) for line in DATA_PATH.read_text(encoding='utf-8').splitlines() if line.strip()]
 9 assert len(rows) == 400
10 assert all(tuple(turn['role'] for turn in row['turns']) == ('system', 'user', 'assistant', 'user', 'assistant') for row in rows)
11 validation = subprocess.run([os.sys.executable, str(ROOT / 'scripts/validate_train.py'), str(DATA_PATH.parent)], cwd=ROOT, text=True, capture_output=True)
12 print(validation.stdout.strip())
13 if validation.returncode != 0:
14     print(validation.stderr)
15     raise RuntimeError('Training pool validation failed.')
16 manifest = {line.split(None, 1)[1].lstrip('*').strip(): line.split(None, 1)[0] for line in MANIFEST_PATH.read_text().splitlines() if len(line.split(None, 1)) == 2}
17 assert manifest.get('dialogues.jsonl') == sha256(DATA_PATH)
18 print('Records:', len(rows))
19 print('Families:', dict(sorted(Counter(row['family'] for row in rows).items())))
20 print('Training data and manifest gates: PASS')
```

- **1:** Defines a SHA-256 helper.
- **2:** Creates a SHA-256 digest object.
- **3:** Opens the file as bytes (`rb`), because hashes are of bytes, not decoded
  characters.
- **4:** Reads one MiB at a time until an empty byte string marks end-of-file.
- **5:** Feeds each chunk into the digest.
- **6:** Returns the final hexadecimal fingerprint.
- **7:** Blank separator.
- **8:** Reads non-empty JSONL lines and parses each into a Python dictionary.
- **9:** Requires exactly 400 records.
- **10:** Requires the exact five-role sequence expected by the later formatting
  and response-mask assumptions.
- **11:** Runs the repository's full training-pool validator with the same Python
  executable as the notebook, from the repository root, capturing output.
- **12:** Prints the validator's normal output.
- **13:** Checks its exit status.
- **14:** Prints validator errors when validation failed.
- **15:** Stops rather than training on invalid data.
- **16:** Parses the checksum manifest into a mapping from filename to digest.
- **17:** Compares the manifest's digest for `dialogues.jsonl` with a fresh hash.
- **18:** Prints the number of loaded records.
- **19:** Counts and prints the four family distribution values.
- **20:** Prints the data gate receipt.

### Cell 8: convert to a Hugging Face dataset

```python
 1 from datasets import Dataset
 2
 3 train_dataset = Dataset.from_list([{'id': row['id'], 'messages': row['turns']} for row in rows]).shuffle(seed=SEED)
 4 print('Dataset columns:', train_dataset.column_names)
 5 print('Dataset rows:', len(train_dataset))
 6 assert train_dataset.column_names == ['id', 'messages']
```

- **1:** Imports Hugging Face Datasets' in-memory `Dataset` type.
- **2:** Blank separator.
- **3:** Converts each repository record into an `id` plus `messages`. The
  conversation content is not rewritten: `messages` is just the standard name
  expected by chat tooling. `.shuffle(seed=SEED)` changes order reproducibly.
- **4:** Prints the column names.
- **5:** Prints the row count.
- **6:** Requires that conversion produced exactly the expected two columns.

### Cell 10: load Gemma through Unsloth

```python
 1 from eval.judge import TUTOR_SYSTEM_PROMPT
 2 from unsloth import FastModel
 3 from unsloth.chat_templates import get_chat_template
 4
 5 def load_model():
 6     model, tokenizer = FastModel.from_pretrained(
 7         model_name=MODEL_ID,
 8         revision=MODEL_REVISION,
 9         dtype=None,
10         max_seq_length=MAX_SEQ_LENGTH,
11         load_in_4bit=True,
12         full_finetuning=False,
13     )
14     tokenizer = get_chat_template(tokenizer, chat_template='gemma-4-thinking')
15     return model, tokenizer
16
17 if RUN_MODE == 'inspect':
18     print('Inspect mode: model loading skipped.')
19 else:
20     model, tokenizer = load_model()
21     print('4-bit Gemma 4 model loaded through Unsloth')
```

- **1:** Imports the canonical tutor system prompt used by evaluation and
  preflight probes.
- **2:** Imports Unsloth's model loader.
- **3:** Imports Unsloth's chat-template selector.
- **4:** Blank separator.
- **5:** Defines one reusable model-loading function.
- **6:** Calls Unsloth's loader and receives the model plus tokenizer/processor.
- **7:** Supplies the pinned model ID.
- **8:** Supplies the immutable model revision.
- **9:** Lets Unsloth choose the appropriate dtype automatically (`None`).
- **10:** Sets Unsloth's sequence-length planning to 1,024.
- **11:** Requests 4-bit loading. This is the “Q” in QLoRA.
- **12:** Explicitly chooses adapter fine-tuning rather than full fine-tuning.
- **13:** Closes the loader call.
- **14:** Installs the `gemma-4-thinking` chat template recommended for the larger
  Gemma 4 models. A template turns role/content dictionaries into the exact
  special-token text the model expects.
- **15:** Returns both objects.
- **16:** Blank separator.
- **17:** Checks for CPU review mode.
- **18:** Reports that model loading was intentionally skipped.
- **19:** Otherwise branch.
- **20:** Loads the model and configured tokenizer for smoke/full mode.
- **21:** Prints the successful-load receipt.

### Cell 11: generation preflight and cleanup

```python
 1 def clear_cuda() -> None:
 2     if torch.cuda.is_available():
 3         torch.cuda.empty_cache()
 4         torch.cuda.reset_peak_memory_stats()
 5
 6 def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 128) -> str:
 7     # Gemma 4's multimodal processor rejects plain text message content; use its text tokenizer for text-only probes.
 8     text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
 9     inputs = text_tokenizer.apply_chat_template(
10         messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt'
11     ).to('cuda')
12     with torch.inference_mode():
13         output = model.generate(
14             **inputs, max_new_tokens=max_new_tokens, use_cache=True,
15             temperature=1.0, top_p=0.95, top_k=64,
16         )
17     generated = output[0, inputs['input_ids'].shape[-1]:]
18     return tokenizer.decode(generated, skip_special_tokens=True).strip()
19
20 if RUN_MODE == 'inspect':
21     print('Inspect mode: base preflight skipped.')
22 else:
23     fixtures = [json.loads(line) for line in FIXTURE_PATH.read_text().splitlines() if line.strip()]
24     for case in fixtures:
25         messages = [{'role': 'system', 'content': TUTOR_SYSTEM_PROMPT}, {'role': 'user', 'content': case['learner_turns'][0]}]
26         print(f'\n--- {case["id"]} ---\n{generate_reply(model, tokenizer, messages)}')
27     print('Untuned base-model preflight: COMPLETE — review outputs above for leakage and utility')
```

- **1:** Defines a CUDA cleanup helper returning nothing.
- **2:** Only performs CUDA calls when CUDA exists.
- **3:** Returns unused cached allocator blocks to PyTorch's allocator. It does
  not shrink the model or manufacture VRAM.
- **4:** Resets the peak counters so later memory reporting starts fresh.
- **5:** Blank separator.
- **6:** Defines a text-generation helper with a default of 128 new tokens.
- **7:** Documents an important Gemma implementation detail: the returned
  object may be a multimodal processor, while these probes contain plain text.
- **8:** Extracts the underlying text tokenizer when there is one; otherwise
  uses the supplied object directly.
- **9:** Starts applying the model's chat template.
- **10:** Adds an assistant-generation prompt, tokenizes, asks for a dictionary
  of tensors, and asks PyTorch for tensor outputs.
- **11:** Moves those tensors to CUDA.
- **12:** Disables gradient tracking during generation because generation is not
  training.
- **13:** Starts the model's autoregressive generation call.
- **14:** Expands the input tensor dictionary, caps new output at the requested
  length, and explicitly enables the KV cache for inference.
- **15:** Uses Unsloth/Gemma's recommended sampling settings: temperature 1.0,
  nucleus probability 0.95, and top-k 64. This is a preflight probe, not the
  benchmark's temperature-0 comparison.
- **16:** Closes generation.
- **17:** Removes the input prompt portion from the returned sequence, leaving
  only newly generated token IDs.
- **18:** Decodes those IDs, removes special tokens, and trims surrounding
  whitespace.
- **19:** Blank separator.
- **20:** Checks inspect mode.
- **21:** Reports that no base-model reply will be generated.
- **22:** Otherwise branch.
- **23:** Loads every fixture JSONL row. Only the first learner turn is used
  below; this is intentionally not the full 48-case evaluation.
- **24:** Iterates over the fixture cases.
- **25:** Builds a two-message prompt: canonical system instruction plus the
  fixture's first learner message.
- **26:** Generates and prints one labeled reply per fixture.
- **27:** Says the human must review the three outputs for obvious leakage,
  formatting failures, or unusable behavior. “COMPLETE” is not a quality score.

### Cell 13: render every conversation as Gemma text

```python
 1 def formatting_prompts_func(examples):
 2     texts = [
 3         tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
 4         for messages in examples['messages']
 5     ]
 6     return {'text': texts}
 7
 8 if RUN_MODE == 'inspect':
 9     print('Inspect mode: formatting requires the tokenizer and is skipped.')
10 else:
11     train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=['messages'])
12     text_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
13     token_lengths = [len(text_tokenizer(text, add_special_tokens=False)['input_ids']) for text in train_dataset['text']]
14     assert all(isinstance(text, str) and text for text in train_dataset['text'])
15     print('Token length min/max:', min(token_lengths), max(token_lengths))
16     assert max(token_lengths) <= MAX_SEQ_LENGTH, 'A training record would be truncated.'
17     print('Formatted example:')
18     print(train_dataset[0]['text'])
19     print('Gemma conversation formatting gate: PASS')
```

- **1:** Defines the batched formatting function expected by Datasets `.map()`.
- **2:** Starts a list of formatted strings.
- **3:** Applies Gemma's template to each conversation without tokenizing yet,
  does not add a generation prompt because the record already contains answers,
  and removes the leading `<bos>` token because the training processor adds it.
- **4:** Completes the list comprehension over the batch's `messages` values.
- **5:** Blank separator inside the expression.
- **6:** Returns a new `text` column.
- **7:** Blank separator.
- **8:** Checks whether only inspection was requested.
- **9:** Explains why formatting is skipped without a loaded tokenizer.
- **10:** Otherwise branch.
- **11:** Applies the function to the whole dataset in batches and removes the
  original structured `messages` column. The resulting dataset retains `id` and
  gains `text`.
- **12:** Gets the plain text tokenizer for length measurement.
- **13:** Tokenizes each formatted string without adding special tokens and
  records its token count.
- **14:** Requires every produced text value to be a non-empty string.
- **15:** Prints the shortest and longest formatted sequence.
- **16:** Refuses to continue if any record would exceed the 1,024-token cap.
- **17:** Labels the human-readable sample output.
- **18:** Prints the first formatted training string so role markers can be seen.
- **19:** Prints the formatting gate receipt.

### Cell 15: attach QLoRA adapters

```python
 1 if RUN_MODE != 'inspect':
 2     model = FastModel.get_peft_model(
 3         model,
 4         finetune_vision_layers=False,
 5         finetune_language_layers=True,
 6         finetune_attention_modules=True,
 7         finetune_mlp_modules=True,
 8         r=LORA_R,
 9         lora_alpha=LORA_ALPHA,
10         lora_dropout=LORA_DROPOUT,
11         bias='none',
12         use_gradient_checkpointing='unsloth',
13         random_state=SEED,
14         use_rslora=False,
15     )
16     model.config.use_cache = False
17     model.print_trainable_parameters()
18     trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
19     assert trainable, 'Unsloth did not expose trainable adapter parameters.'
20     print('QLoRA adapter injection gate: PASS')
```

- **1:** Skips adapter creation in inspect mode.
- **2:** Calls Unsloth's PEFT/LoRA setup and replaces `model` with the patched
  trainable model.
- **3:** Passes the loaded base model into the setup call.
- **4:** Keeps vision parameters/adapters off.
- **5:** Enables language-layer adaptation.
- **6:** Enables attention projections.
- **7:** Enables MLP projections.
- **8:** Supplies rank 16.
- **9:** Supplies alpha 32.
- **10:** Supplies zero dropout.
- **11:** Chooses not to train bias parameters.
- **12:** Selects Unsloth's memory-saving gradient-checkpointing path.
- **13:** Gives Unsloth the run seed for adapter initialization and related
  randomized setup.
- **14:** Keeps rank-stabilized LoRA disabled.
- **15:** Closes the setup call.
- **16:** Disables the model's inference KV cache for training. Caching old
  activations is incompatible with the usual memory-saving checkpointed training
  flow.
- **17:** Prints trainable parameter counts.
- **18:** Collects every parameter that is allowed to receive gradients.
- **19:** Refuses to continue if the adapter setup accidentally exposed nothing.
- **20:** Prints the injection gate receipt. This does not yet update a weight.

### Cell 17: configure SFT and response-only labels

```python
 1 from trl import SFTConfig, SFTTrainer
 2 from unsloth.chat_templates import train_on_responses_only
 3
 4 if RUN_MODE != 'inspect':
 5     output_dir = Path(tempfile.mkdtemp(prefix='socratic-gemma4-smoke-')) if RUN_MODE == 'smoke' else FINAL_ADAPTER_DIR
 6     training_args = SFTConfig(
 7         output_dir=str(output_dir),
 8         dataset_text_field='text',
 9         num_train_epochs=EPOCHS,
10         max_steps=1 if RUN_MODE == 'smoke' else -1,
11         per_device_train_batch_size=MICRO_BATCH_SIZE,
12         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
13         learning_rate=LEARNING_RATE,
14         lr_scheduler_type='cosine',
15         warmup_steps=5,
16         optim='adamw_8bit',
17         weight_decay=0.001,
18         max_grad_norm=0.3,
19         bf16=True,
20         tf32=True,
21         gradient_checkpointing=True,
22         logging_steps=1 if RUN_MODE == 'smoke' else 10,
23         report_to='none',
24         save_strategy='no',
25         eval_strategy='no',
26         seed=SEED,
27         data_seed=SEED,
28         max_length=MAX_SEQ_LENGTH,
29         packing=False,
30         dataset_num_proc=1,
31         run_name='socratic-gemma4-31b-qlora',
32     )
33     trainer = SFTTrainer(
34         model=model, tokenizer=tokenizer, train_dataset=train_dataset, args=training_args
35     )
36     trainer = train_on_responses_only(
37         trainer, instruction_part='<|turn>user\n', response_part='<|turn>model\n'
38     )
39     labels = trainer.train_dataset[0]['labels']
40     assert any(label == -100 for label in labels), 'Instruction tokens were not masked.'
41     assert any(label != -100 for label in labels), 'No assistant tokens remain for loss.'
42     print('Response-only loss mask gate: PASS')
```

- **1:** Imports TRL's supervised fine-tuning configuration and trainer.
- **2:** Imports Unsloth's response-only masking helper.
- **3:** Blank separator.
- **4:** Builds the expensive training objects only outside inspect mode.
- **5:** Uses a temporary directory for smoke mode and `train/adapter` for full
  mode. `mkdtemp` creates the temporary directory immediately.
- **6:** Starts the SFT configuration.
- **7:** Gives the trainer its output location.
- **8:** Tells the trainer to read its already formatted examples from `text`.
- **9:** Supplies the three-epoch request.
- **10:** Overrides the epoch plan in smoke mode with one optimizer step; `-1`
  means no step limit in full mode.
- **11:** Sets one example per device micro-batch.
- **12:** Accumulates four micro-batches per optimizer update.
- **13:** Sets the learning rate.
- **14:** Selects cosine learning-rate decay.
- **15:** Uses five warmup optimizer steps.
- **16:** Selects 8-bit AdamW.
- **17:** Applies weight decay of 0.001.
- **18:** Clips gradient norm at 0.3.
- **19:** Enables BF16 math.
- **20:** Enables NVIDIA TF32 acceleration where PyTorch uses it.
- **21:** Enables gradient checkpointing.
- **22:** Logs every step in smoke mode and every ten steps in full mode.
- **23:** Does not send metrics to an external tracker such as Weights & Biases.
- **24:** Does not save intermediate checkpoints.
- **25:** Does not run evaluation during training.
- **26:** Sets the trainer's random seed.
- **27:** Sets the dataset-order seed separately.
- **28:** Sets the 1,024-token maximum.
- **29:** Disables packing, so separate conversations are not concatenated.
- **30:** Uses one dataset preprocessing worker, keeping the path simple and
  less prone to worker/process surprises.
- **31:** Gives the run a human-readable name.
- **32:** Closes `SFTConfig`.
- **33:** Starts constructing TRL's SFT trainer.
- **34:** Supplies the patched model, tokenizer, formatted dataset, and all
  settings. Constructing a trainer is not training yet.
- **35:** Closes trainer construction.
- **36:** Wraps the trainer so loss is computed on responses only.
- **37:** Tells the wrapper how Gemma marks user/instruction and model/response
  turns. These strings must match the rendered training format.
- **38:** Closes the wrapper call.
- **39:** Reads the first example's generated labels.
- **40:** Requires at least one `-100` ignored instruction label.
- **41:** Requires at least one non-ignored assistant label.
- **42:** Prints the response-mask gate receipt.

### Cell 18: the first actual weight-changing call

```python
 1 if RUN_MODE == 'inspect':
 2     print('Inspect mode: optimizer step skipped.')
 3 else:
 4     started = time.time()
 5     train_result = trainer.train()
 6     elapsed_seconds = time.time() - started
 7     metrics = dict(train_result.metrics)
 8     assert torch.isfinite(torch.tensor(float(metrics['train_loss'])))
 9     print('Training returned in minutes:', f'{elapsed_seconds / 60:.1f}')
10     print(json.dumps(metrics, indent=2, default=str))
11     print('Peak allocated GiB:', f'{torch.cuda.max_memory_allocated() / 2**30:.2f}')
12     print('Peak reserved GiB:', f'{torch.cuda.max_memory_reserved() / 2**30:.2f}')
13     print('Training loss and memory gate: PASS')
```

- **1:** Checks whether training was intentionally skipped.
- **2:** Reports that no optimizer step occurred in inspect mode.
- **3:** Otherwise branch.
- **4:** Records the start time.
- **5:** Calls TRL's training loop. This is the first line in the notebook that
  performs forward passes, computes loss, backpropagates, and updates LoRA
  weights.
- **6:** Computes elapsed wall-clock time.
- **7:** Copies the returned metrics into a regular dictionary.
- **8:** Requires the recorded training loss to be finite, rejecting NaN or
  infinity. A finite loss is necessary but says nothing by itself about useful
  tutor behavior.
- **9:** Prints elapsed minutes.
- **10:** Prints metrics such as loss, runtime, samples, and steps in readable
  JSON.
- **11:** Prints the largest amount of GPU memory actually allocated by tensors.
- **12:** Prints the largest amount reserved by PyTorch's caching allocator;
  reserved and allocated are different numbers.
- **13:** Prints the training gate receipt.

### Cell 19: heading for saving

This Markdown cell says smoke saves to temporary output while full mode saves the
adapter and evidence. Saving is not a second training method; it makes the
learned adapter usable outside the trainer process.

### Cell 20: save adapter and full evidence

```python
 1 def write_manifest(directory: Path, output: Path) -> None:
 2     files = sorted(path for path in directory.rglob('*') if path.is_file())
 3     output.write_text(''.join(f'{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n' for path in files))
 4
 5 if RUN_MODE == 'inspect':
 6     print('Inspect mode: save skipped.')
 7 else:
 8     trainer.save_model(str(output_dir))
 9     tokenizer.save_pretrained(str(output_dir))
10     if RUN_MODE == 'smoke':
11         print('Temporary smoke adapter saved:', sorted(path.name for path in output_dir.iterdir()))
12     else:
13         config = {
14             'model': {'id': MODEL_ID, 'revision': MODEL_REVISION, 'load_in_4bit': True, 'max_seq_length': MAX_SEQ_LENGTH},
15             'dataset': {'path': str(DATA_PATH.relative_to(ROOT)), 'sha256': sha256(DATA_PATH), 'records': len(rows)},
16             'seed': SEED,
17             'qlora': {'backend': 'unsloth', 'rank': LORA_R, 'alpha': LORA_ALPHA, 'target_modules': 'all-linear', 'dropout': LORA_DROPOUT},
18             'training': {'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'scheduler': 'cosine', 'micro_batch_size': MICRO_BATCH_SIZE, 'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS, 'max_length': MAX_SEQ_LENGTH, 'assistant_only_loss': True, 'optimizer': 'adamw_8bit', 'final_checkpoint_only': True},
19             'runtime': {'python': os.sys.version.split()[0], 'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'gpu': gpu_name, 'gpu_vram_gib': gpu_vram, 'system_ram_gib': total_ram, 'packages': {name: package_version(name) for name in ('unsloth', 'unsloth-zoo', 'bitsandbytes', 'transformers', 'trl', 'datasets', 'accelerate')}},
20         }
21         import yaml
22         FINAL_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
23         FINAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
24         (ROOT / 'train/config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
25         (FINAL_LOG_DIR / 'notebook-log.json').write_text(json.dumps({'metrics': metrics, 'config': config}, indent=2, default=str))
26         environment = subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True).stdout
27         (FINAL_LOG_DIR / 'environment.txt').write_text(environment)
28         write_manifest(FINAL_ADAPTER_DIR, ROOT / 'train/adapter.sha256')
29         print('Final adapter and evidence saved.')
```

- **1:** Defines a helper that writes a file manifest and returns nothing.
- **2:** Recursively finds every regular file under the adapter directory and
  sorts the list for stable ordering.
- **3:** Writes each file's SHA-256 and repository-relative path, one per line.
- **4:** Blank separator.
- **5:** Checks inspect mode.
- **6:** Reports that no adapter is saved.
- **7:** Otherwise branch.
- **8:** Serializes the trainable adapter through the trainer's save method.
- **9:** Saves the tokenizer/template alongside it. The adapter needs compatible
  tokenization at reload time.
- **10:** Checks smoke mode.
- **11:** Prints the names of temporary files so serialization can be seen.
- **12:** Otherwise this is full mode.
- **13:** Starts the configuration snapshot dictionary.
- **14:** Records model identity, 4-bit loading, and context limit.
- **15:** Records the dataset path, its current digest, and row count.
- **16:** Records the seed.
- **17:** Records the Unsloth QLoRA backend and adapter settings. The snapshot
  describes the selected target policy as `all-linear`; the actual injection is
  performed by Unsloth's language/attention/MLP flags on line 4–7 above.
- **18:** Records the important training choices, including response-only loss
  and final-checkpoint-only intent.
- **19:** Records Python, PyTorch, CUDA, hardware, RAM, and package versions.
- **20:** Closes the snapshot dictionary.
- **21:** Imports YAML only when a full artifact is actually being written.
- **22:** Creates the final adapter directory if needed.
- **23:** Creates the log directory if needed.
- **24:** Writes the effective configuration as `train/config.yaml`.
- **25:** Writes metrics and config as `train/logs/notebook-log.json`.
- **26:** Runs `pip freeze` with the current interpreter, captures it, and
  requires that the command succeed.
- **27:** Writes the resolved package environment to `train/logs/environment.txt`.
- **28:** Hashes every adapter file into `train/adapter.sha256`.
- **29:** Prints the full-artifact receipt.

### Cell 21: after-training instructions

This final Markdown cell tells you what the notebook intentionally does not
claim. The full run is not the evaluation. After it completes, run the separate
48-case benchmark twice:

1. pinned Gemma base + canonical prompt, no adapter;
2. the same pinned base + the saved adapter.

Keep cases, prompt, tokenizer/template, decoding, and judge fixed. Compare
leakage and actionable diagnosis. Do not select a checkpoint because it looks
best: this experiment intentionally has one final adapter and no validation-set
checkpoint competition.

---

## 7. What the smoke output means

Read the smoke output in this order.

### Hardware and dependency gate

Good signs:

- the expected GPU is shown;
- VRAM is around the intended 24 GiB card and at least the notebook's 22 GiB
  threshold;
- BF16 is `True`;
- Unsloth and bitsandbytes have versions rather than `missing`.

A failure here means the environment is not the declared experiment. Do not
“fix” it by silently switching to CPU, FP16, another model, or a different
quantization mode. That would be a new experiment.

### Training-data and manifest gates

Good signs:

- `Records: 400`;
- the expected family counts appear;
- the validator says the pool passed;
- the expected and actual SHA-256 values match.

A checksum mismatch means the bytes on disk are not the committed data the
experiment claims to use.

### Formatting gate

Inspect the printed formatted example. You should see Gemma role boundaries and
an assistant/model turn. Check the min/max token lengths; the maximum must be no
more than 1,024.

### QLoRA injection gate

The trainable parameter report should show a small trainable fraction compared
with the entire 31B model. This proves the intended adapter path was exposed; it
does not prove that training quality will be good.

### Response-only loss gate

The gate must show that some labels are masked (`-100`) and some remain as
assistant targets. If either assertion fails, stop. Training with the wrong mask
would teach the model a different objective.

### Training loss and memory gate

The loss must be finite. A finite one-step loss means the forward/backward/update
pipeline completed without NaN/Inf. It is not a pass/fail score for tutor
behavior, and one step is far too little to show learning.

Peak **allocated** and **reserved** memory tell you whether the real recipe fit.
Do not compare only to a theoretical model-size number: activations,
quantization buffers, optimizer state, CUDA workspaces, and framework overhead
all matter.

### Temporary adapter save

Smoke should list adapter files, including adapter weights and configuration.
That proves the output can be serialized. The result is temporary and is not the
trained artifact you should benchmark.

---

## 8. What full mode will produce

A successful full run is intended to leave:

```text
train/adapter/             # LoRA adapter plus tokenizer files
train/config.yaml          # effective experiment snapshot
train/seed                 # one-line seed receipt
train/logs/notebook-log.json
train/logs/environment.txt
train/adapter.sha256       # hashes of every adapter file
```

The adapter directory is much smaller than the 31B base model because it stores
the learned LoRA changes rather than a full copy of Gemma. At inference time,
load the same base revision and attach this adapter.

### Important operational caveats in the current notebook

- The notebook refuses full mode unless you explicitly confirm it, but the
  current Gemma notebook does not itself check that `train/adapter/` is empty.
  Before a new final-only run, move any old adapter directory aside so files from
  two runs cannot be mixed.
- Smoke output uses a temporary directory created with `tempfile.mkdtemp`. The
  notebook does not explicitly remove that directory at the end of Cell 20; it
  is disposable temporary output, not a repository artifact.
- The notebook does not perform a validation split or mid-training evaluation.
  There is no checkpoint winner. The only saved result is the final adapter.
- The three preflight generations use `temperature=1.0`, `top_p=0.95`, and
  `top_k=64`, as recommended for the Gemma 4 generation probe. The separate
  benchmark's decoding contract is its own fixed setting; do not treat these
  three replies as benchmark scores.
- “Training loss went down” is not equivalent to “leakage decreased” or
  “diagnosis improved.” Only the fixed external benchmark can support those
  claims.

---

## 9. A safe full-run checklist

Before changing the two mode values:

1. Confirm you are opening `train/gemma4_qlora_guided.ipynb`.
2. Confirm the kernel is the dedicated Python 3.12 `socratic-train` environment.
3. Read the printed GPU, VRAM, BF16, and package versions.
4. Confirm the 400-record count, family balance, validator result, and matching
   SHA-256 digest.
5. Read the three untouched base-model replies.
6. Confirm the rendered Gemma example is non-empty and no longer than 1,024
   tokens.
7. Confirm the QLoRA trainable-parameter report is present.
8. Confirm response-only masking has both ignored and active labels.
9. Confirm peak smoke memory leaves enough headroom; a pass close to the limit
   is still a tight fit.
10. Move any existing `train/adapter/` aside.
11. Restart the kernel. This prevents smoke model/trainer state from leaking
    into full mode.
12. Set `RUN_MODE = 'full'` and `CONFIRM_FULL_RUN = True`.
13. Run top to bottom, one cell at a time.
14. Do not interrupt the run unless necessary; if interrupted, treat the output
    as incomplete and do not call it the final artifact.
15. After full mode, inspect `train/config.yaml`, logs, adapter manifest, and
    the independent reload reply.
16. Run the separate 48-case benchmark against both base and adapter.

---

## 10. What this experiment can and cannot conclude

### It can establish

- the exact pinned Gemma artifact can load through the supported Unsloth path;
- the committed training pool is the input;
- the tokenizer/template produces bounded text;
- the adapter is attached to intended text-model components;
- response-only labels exist;
- the declared one-step or three-epoch SFT run completes with finite loss;
- the resulting adapter can be saved and reloaded.

### It cannot establish by itself

- that the adapter is better than the base model;
- that the model never leaks completed solutions;
- that the model diagnoses misconceptions usefully;
- that the model generalizes beyond the synthetic training pool;
- that the chosen rank, learning rate, or epoch count is optimal;
- that the result is bit-for-bit identical on every CUDA/software stack;
- that a QLoRA result is interchangeable with the old BF16 Qwen LoRA result.

Those stronger claims require the separate benchmark, fixed comparison controls,
and careful interpretation of its logs.

---

## 11. Short glossary

- **Adapter:** the small learned file added to a frozen base model.
- **Assistant-only loss:** only assistant/model response tokens contribute to
  the training error.
- **Base model:** the original pretrained Gemma model before this adapter.
- **BF16:** a 16-bit floating-point computation format.
- **CUDA:** NVIDIA's GPU programming platform used by PyTorch.
- **Epoch:** one complete pass through the training pool.
- **Gradient:** a number describing how a parameter should change to reduce
  loss.
- **Gradient accumulation:** adding gradients from multiple micro-batches before
  changing parameters.
- **Gradient checkpointing:** recomputing intermediate values to save VRAM.
- **LoRA:** low-rank trainable updates attached to frozen model layers.
- **Loss:** a numeric measure of how poorly the model predicted the target
  tokens; lower is generally better for the training objective.
- **Micro-batch:** the examples processed together in one forward/backward pass.
- **Optimizer:** the update rule that changes trainable parameters from
  gradients.
- **QLoRA:** LoRA training with a quantized, usually 4-bit, frozen base model.
- **SFT:** supervised fine-tuning from examples with known desired responses.
- **Token:** a model-specific piece of text represented by an integer ID.
- **Tokenizer:** the converter between text/chat messages and token IDs.
- **Training step:** in this notebook, an optimizer update; four micro-batches
  contribute to one such update in full mode.

---

## 12. Primary references in this repository

- Canonical notebook: [`gemma4_qlora_guided.ipynb`](gemma4_qlora_guided.ipynb)
- Environment and run order: [`README.md`](README.md)
- Training data contract: [`../data/train/README.md`](../data/train/README.md)
- Training-pool validator: [`../scripts/validate_train.py`](../scripts/validate_train.py)
- Canonical tutor prompt and judge: [`../eval/judge.py`](../eval/judge.py)
- Historical Qwen comparison: [`../docs/research/lora-training-stack.md`](../docs/research/lora-training-stack.md)
- Official Gemma 4/Unsloth guide: <https://unsloth.ai/docs/models/gemma-4/train>
