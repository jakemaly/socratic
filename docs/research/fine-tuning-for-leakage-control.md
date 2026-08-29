# Fine-tuning for leakage control

**Scope.** This report treats leakage as the assistant emitting a completed code solution, exact requested output, formula, built-in/API name, or confirmation of an algorithm. The target is not a keyword filter: it is a useful Socratic response that refuses the artifact while giving a diagnostic next step.

## What the current result says

The repository has 400 SFT conversations, assistant-only loss, and a 31B Gemma 4 QLoRA recipe. V2 scores **89.58%** on the fixed 48-case judge benchmark, but leakage is **6.25% (3/48)** versus V1 **4.17% (2/48)**. That is one case, or 2.08 percentage points—not evidence that V2 is reliably worse. Keep the benchmark fixed and report counts beside percentages; treat the leakage change as a signal for targeted ablations, not a reason to chase training loss.

The likely control lever is target behavior in the assistant completions. Gemma’s official tuning guide recommends a diverse task-specific dataset, conversational `messages`, and QLoRA: 4-bit frozen base weights plus trainable LoRA adapters ([Google Gemma QLoRA guide](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)). TRL likewise supports conversational data, assistant-only loss, completion-only loss, and packing ([TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)).

## SFT data: first and best experiment

**Balance behavior, not merely topics.** Partition the 400 examples by the five prohibited artifact classes, prompt intent (direct demand, indirect demand, “just confirm,” debugging, and follow-up pressure), and useful alternative (question, observation/check, test design, or conceptual explanation). Make these cells visible in the manifest. Oversampling a rare leakage class is reasonable, but preserve prompt diversity and keep a held-out paraphrase slice. Repeating identical refusals risks teaching a brittle phrase rather than the boundary.

**Use hard negatives as prompts, not targets.** Add or rewrite examples where the user asks for the forbidden artifact and the assistant gives a concise boundary plus an actionable diagnostic. Include near-misses: partial code, a single formula term, a named built-in, “is this algorithm correct?”, and multi-turn escalation. Do not put the completed artifact in the assistant target merely to teach what *not* to say; SFT would reward reproducing it. If negative responses are available, use them for preference pairs later.

**Preserve assistant-only loss.** It concentrates gradient on the desired response and avoids teaching the model to imitate user text. Verify the rendered Gemma chat template and the labels: every assistant token intended for learning must be unmasked, and user/system tokens must be `-100`. TRL documents that padding is ignored and that assistant-only masking depends on the conversational template; Transformers documents using the model’s chat template rather than hand formatting ([Transformers chat templates](https://huggingface.co/docs/transformers/en/chat_templating)). This is a correctness gate, not a leakage remedy by itself.

**Do not enable packing yet.** Packing improves token efficiency, but on only 400 short conversations the gain is less valuable than an auditable mask. TRL’s packing and truncation options are real alternatives, not automatically equivalent recipes. Keep `packing=False` for the next behavior ablation; enable it only after comparing per-token masks and a short canary’s outputs.

**Curriculum is optional, not the first fix.** A defensible small experiment is clear refusal-plus-diagnosis examples first, followed by indirect and adversarial paraphrases, while retaining a replay mixture of ordinary tutoring. The tradeoff is stronger boundary learning versus reduced general tutoring and order effects. There is no Gemma/TRL guarantee that this curriculum improves leakage; compare it against shuffled training under identical steps and seeds.

## Preference optimization: useful only after SFT is clean

Preference training can make the distinction explicit: for each prompt, chosen is “refuse artifact + useful next step,” rejected is “leaks artifact” or “refuses without help.” Keep pairs contrastive and category-balanced. Do not use the 48 judge cases verbatim; create paraphrases and new values so the test remains an evaluation surface.

* **DPO** optimizes the chosen/rejected probability margin relative to a reference model. The original paper presents it as a classification-like alternative to RLHF ([Rafailov et al., DPO](https://arxiv.org/abs/2305.18290)); TRL requires preference-format data and supports precomputing reference log probabilities ([TRL DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)). It is conceptually direct, but a 31B reference-policy computation is the least comfortable choice on one 24 GiB card. Precompute reference scores, use micro-batch 1 and short sequences, and treat this as a later experiment—not the default.
* **ORPO** combines SFT with an odds-ratio preference penalty and removes the reference model ([Hong et al., ORPO](https://arxiv.org/abs/2403.07691); [TRL ORPO](https://huggingface.co/docs/trl/en/orpo_trainer)). It is a plausible one-GPU option, but it still needs reliable pairs and an extra loss weight; its published evidence is not specific to this tutor or Gemma 4.
* **SimPO** uses average sequence log-probability as a reference-free reward and adds a target margin ([Meng et al., SimPO](https://arxiv.org/abs/2405.14734)). It reduces reference-model memory, but margin/beta tuning and response-length effects become new variables. Prefer it over DPO only if an ORPO/SFT ablation shows a real preference-data ceiling.

None of these methods guarantees “never leak.” They optimize relative likelihood, so bad pair construction can teach terse refusals, reward verbosity, or suppress useful explanations. Keep assistant-only response masking where the trainer supports it, and evaluate the actual generated text.

## Hardware and execution recommendation

Unsloth’s first-party Gemma 4 guide explicitly reports approximately **22 GB** for 31B QLoRA, recommends batch size 1/reduced sequence length when OOM occurs, and shows response-only training ([Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train.md)). Thus the existing pre-quantized 4-bit path, micro-batch 1, gradient accumulation, short 1,024-token cap, and Unsloth gradient checkpointing are feasible but tight on a 24 GiB RTX 3090. Full fine-tuning is out of scope. Preference training adds compute and, for DPO, reference inference; do not increase rank, context, or batch simultaneously with a new objective.

## Evaluation protocol

For every candidate, run the same 48 prompts, system prompt, model revision, adapter, decoding settings, and judge. Save raw outputs and classify leakage by artifact type, plus actionable diagnosis and utility. Report `leaks / 48`, `useful / 48`, and the full per-case table; one case moves the rate by 2.08 points. Add a small adversarial holdout with paraphrases and multi-turn pressure, but do not replace the fixed benchmark. Compare V1, V2, and each single-variable ablation; retain the candidate only when leakage falls without a material utility regression. The immediate next run should be balanced hard-negative SFT with assistant-only loss and packing disabled; preference optimization is the second-stage experiment if that does not lower the three V2 leakage cases.

## Sources

All sources above are first-party documentation or original papers: Google Gemma, Hugging Face Transformers/TRL, Unsloth, and the authors’ original DPO/ORPO/SimPO papers. No secondary summaries were used.
