# Release surface audit

This inventory describes the tracked tree for the research and evaluation
release. Model weights, raw training inputs, and interactive frontend materials
are intentionally kept out of the shipping tree.

## Retain

- The evaluation harness, schemas, scripts, tests, benchmark cases, and
  published evidence.
- Research and user-facing documentation needed to explain the project and its
  scoped benchmark claim.
- Small training receipts and validation code that explain the selected run
  without shipping its raw inputs, notebooks, environment, or weights.

## Archived from the release tree

- Raw pools under `data/train/`, `data/train-v2/`, `data/train-v3/`, and
  `data/train-next/`, including provenance, manifests, and generated dialogue
  records.
- Training notebooks and the notebook-specific walkthrough.
- The training requirements file and generated adapter checksum receipt.
- Superseded Qwen-stack and V3 planning notes whose conclusions are preserved
  in the consolidated training-evolution record.
- Generated planning HTML, internal planning mirrors, tracked training-run
  logs, and notebook-run output.

These files remain recoverable from Git history. They are not copied into the
release tree because this repository does not execute training or ship model
weights. The training generator now requires an explicit output directory, and
`.gitignore` blocks raw training pools, notebooks, and local run outputs from
returning accidentally.

## Archive

The removed files are archived in Git history in commits `ced396b`,
`65b4b72`, and `4e461bc`; they can be recovered without keeping copies in the
release tree.

The corresponding ignore rules prevent these local-only paths from returning.
No tracked API keys, model weights, caches, or oversized accidental files were
found during the audit.
