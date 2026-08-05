# Synthetic training pool

This is the 400-dialogue pilot for issue #9. `exercises.jsonl` is the separate
training exercise-pool manifest; it must not be reused by benchmark or gallery
records.

The default generator is deterministic and credential-free:

```bash
python3 scripts/generate_dialogues.py
python3 scripts/validate_train.py data/train
```

Use `--mode live` only after the pinned teacher-model credentials from issue #10
are available. The live path uses the same shared gates in `scripts/gates.py`.

`balance.csv` is the family-by-exercise table. Every family has 100 records and
each of the four exercises contributes 25 records to each family. Rejected
attempts remain in `provenance.jsonl` with one or more of the standard reason
codes: `leaked`, `low-utility`, `malformed`, `duplicate`, or `quota-exceeded`.
`manifest.sha256` covers the versioned JSONL artifacts; the config and balance table are derived metadata.
