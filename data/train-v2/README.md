# Synthetic training pool V2

This candidate preserves the V1 400-record matrix and replaces six records with hard negatives for semantic leakage under answer pressure. It targets exact numeric outputs, complete rule/formula paraphrases, decisive built-in hints, accumulator initialization, and confirmation of algorithmic strategies.

V1 remains at `data/train/`. Before training, point the training notebook at this directory and save the resulting V2 adapter separately. Validate with `python3 scripts/validate_train.py data/train-v2`.
