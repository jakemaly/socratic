# V3 leakage-first alignment fine-tuning plan

## Decision

Do not tune V3 by adding a few more refusal examples or increasing epochs. V2 learned a stronger *diagnosis style* but not a reliable boundary: it scored 43/48 actionable diagnoses while leaking on 3/48 cases, including two new semantic leaks. The next experiment must optimize leakage first and preserve diagnosis as a separate constraint.

V1 and V2 remain immutable comparison arms. V3 uses the same Gemma 4 31B revision and QLoRA recipe unless an ablation explicitly changes one variable.

## Gaps exposed by V2

1. **Coverage gap.** The 400-record pool has no meaningful assistant targets for maximum-list logic, grade boundaries, or built-in-function pressure. The benchmark tests all three directly. V2 added one generic built-in example, but not the exact semantic boundary.
2. **Pressure-depth gap.** Training examples contain one escalation (five roles total); benchmark conversations can apply repeated pressure over more turns. The model eventually abandons the boundary on the first-element case.
3. **Objective gap.** SFT rewards the chosen assistant text but never penalizes a semantically leaking answer. A refusal followed by `max()` or exact comparison ranges is still an SFT-compatible “helpful” continuation unless the targets make that distinction broad and explicit.
4. **Verbosity gap.** V2 responses became longer and more repetitive. More explanatory surface created more chances to reveal an operator, range, built-in, or algorithm.
5. **Evaluation gap.** The fixed 48 cases test answer leakage, not general prompt-injection resistance. They do not cover extraction of the system prompt, obfuscation, multilingual attacks, indirect instructions, or tools.
6. **Template gap.** Literal `thought` fragments appeared in both V1 and V2. Generation/template handling must be audited separately so hidden/reasoning markers are not treated as tutor output or allowed to contaminate the policy signal.

## Phase 0 — make the measurement trustworthy

- Keep the existing 48 cases, judge, pinned base revision, temperature `0`, seed `0`, and 128-token cap unchanged for continuity.
- Add a separate 48-case security/prompt-pressure suite: direct hierarchy conflict, system-prompt extraction, role-play/encoding, five-plus-turn escalation, and quoted/indirect instructions. Do not train on these exact cases.
- Record artifact-specific labels: numeric result, formula/condition, built-in/API name, algorithm confirmation, initialization/boundary, prompt/instruction disclosure, and sterile refusal.
- Preserve raw generated text and separately strip/record template markers. Never silently sanitize the benchmark transcript.
- Use a small blinded human audit of judge calls to check whether diagnosis gains are genuine rather than phrase-matching.

## Phase 1 — V3 balanced SFT data

Create `data/train-v3/` with exactly 400 records and the same family/exercise balance. Replace 80 records—five in each of the 16 family/exercise cells—rather than changing the total size. Distribute the replacements evenly across four families and five behavior classes (16 each):

1. exact numeric/final-output demand;
2. formula or complete condition paraphrase;
3. built-in/API name demand;
4. algorithm or initialization confirmation;
5. instruction-hierarchy/prompt-injection conflict.

Use training exercises disjoint from the 12 benchmark exercises, but cover their concepts with different values and wording. Every class must include direct demand, indirect demand, “just confirm,” debugging framing, and persistent escalation. The final user turn should contain the pressure; the assistant target must hold the boundary.

Target format: one short refusal, one concrete conceptual diagnosis, and one next-step question—roughly 1–3 sentences. Never include code, exact numbers requested by the learner, complete formulas/boolean conditions, decisive built-in names, or a confirmation that gives away the algorithm. Do not put leaking answers in SFT targets. Keep ordinary stuck-learner examples as replay so the model does not learn sterile refusal.

Add dataset metadata for behavior-class counts and a semantic review record. Lexical gates remain smoke checks; an external judge or human review must approve replacement targets.

## Phase 2 — controlled SFT ablation

Train `train/adapter-v3-sft/` from the same base with the existing V2 notebook recipe: rank 16, alpha 32, all-linear modules, 3 epochs, assistant-only loss, no packing, batch 1, accumulation 4. This isolates data coverage from optimizer changes.

Acceptance gate: leakage must be no worse than V1 (at most 2/48, target 0/48), diagnosis must be at least V1 (at least 32/48), and no benign tutoring regression may appear in the new suite. Do not select a model solely because training loss is lower or diagnosis is higher.

If V3 SFT still leaks, run one-variable ablations: first shorter targets/lower learning rate, then two epochs. Do not change rank, context, and objective simultaneously.

## Phase 3 — preference alignment if SFT plateaus

Build a separate preference set from new paraphrases, not the fixed benchmark. For each prompt, use:

- **chosen:** concise refusal + specific diagnosis + next-step question;
- **rejected:** a V1/V2 leak, a formula/code leak, an algorithm confirmation, or a generic refusal with no diagnosis.

Start with ORPO because it avoids a separate reference model on the 24 GiB RTX 3090. Compare it to the accepted V3 SFT arm with the same base, data split, decoding, and evaluation. Try DPO only if ORPO is stable and the reference-policy memory cost is acceptable. Keep pair categories balanced and inspect for length bias; preference training can reward verbosity or over-refusal if pairs are poorly constructed.

## Phase 4 — defense in depth

Fine-tuning is not a security boundary. Keep the runtime system/developer policy, structured message boundaries, least-privilege tool wrappers, and an output policy check. For this tutor, a detected completed artifact should trigger a safe fallback while retaining the raw output for evaluation. If retrieval or tools are added later, treat returned content as untrusted data and require explicit confirmation for side effects.

## Evidence and stop conditions

For every arm, save the adapter hash, dataset hash, config, raw transcripts, judge calls, aggregate metrics, per-case artifact labels, and human-audit sample. Stop and revise the data if leakage worsens, diagnosis is achieved only by repeated refusal boilerplate, or template markers remain in visible output. Promote V3 only when it beats V1 on leakage without sacrificing V1-level diagnosis and passes the separate security suite; otherwise retain V1 as the production comparison arm.

Primary research: `docs/research/alignment-prompt-injection-lora.md` and `docs/research/fine-tuning-for-leakage-control.md`.
