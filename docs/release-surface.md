# Release surface audit

This inventory describes the tracked tree for the local portfolio release. Git
history preserves removed files, so no separate copy of generated material is
needed.

## Retain

- The six-exercise demo, fixture data, evaluation harness, schemas, scripts,
  tests, benchmark cases, and published evidence.
- Research and user-facing documentation needed to explain the project and its
  scoped benchmark claim.
- The canonical Gemma 4 training contract, current training configuration,
  checksum metadata, and source recipes until the training-surface cleanup is
  complete.

## Remove before the final release

- Retired raw training pools and their provenance files.
- Training notebooks, including the older Qwen recipe and superseded Gemma
  experiment notebooks.
- Any remaining training output or generated metadata that is not needed to
  explain the selected result.

This is the next training-surface cleanup ticket; those files remain recoverable
from Git history while that work is pending.

## Removed in the repository-surface audit

- Generated planning HTML.
- Internal planning and ticket mirrors.
- Tracked training-run logs and notebook-run output.

The corresponding ignore rules prevent these local-only paths from returning.
No tracked API keys, model weights, caches, or oversized accidental files were
found during the audit.
