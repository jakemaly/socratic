# Synthetic training pool V3

V3 preserves the V1/V2 400-dialogue matrix and replaces 80 records: five in every family/exercise cell. The replacement set is balanced across numeric-result, formula-or-condition, built-in/API, algorithm-or-initialization, and instruction-hierarchy behaviors. Targets are concise refusal-plus-diagnosis responses; benchmark cases remain held out.

Validate with `python3 scripts/validate_train.py data/train-v3`. The V3 SFT notebook is `train/gemma4_qlora_v3.ipynb`; it writes only to `train/adapter-v3-sft/`.
