from __future__ import annotations

import unittest

from eval.judge import TUTOR_SYSTEM_PROMPT
from scripts.training_contract import (
    CANONICAL_SYSTEM_PROMPT_SHA256,
    canonicalize_turns,
    rows_sha256,
    validate_training_rows,
)


class TrainingContractTests(unittest.TestCase):
    def test_canonicalize_replaces_only_system_prompt(self) -> None:
        turns = [
            {"role": "system", "content": "old persona"},
            {"role": "user", "content": "keep me"},
            {"role": "assistant", "content": "keep this too"},
            {"role": "user", "content": "and this"},
            {"role": "assistant", "content": "last"},
        ]
        result = canonicalize_turns(turns)
        self.assertEqual(result[0]["content"], TUTOR_SYSTEM_PROMPT)
        self.assertEqual(result[1:], turns[1:])
        self.assertEqual(turns[0]["content"], "old persona")

    def test_validation_rejects_prompt_mismatch(self) -> None:
        row = {
            "id": "dialogue-test-001",
            "turns": [
                {"role": "system", "content": "wrong"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer?"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "question?"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "non-canonical system prompt"):
            validate_training_rows([row])

    def test_validation_accepts_canonical_conversation(self) -> None:
        row = {
            "id": "dialogue-test-001",
            "turns": [
                {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "diagnosis?"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "next question?"},
            ],
        }
        self.assertEqual(validate_training_rows([row]), [row])
        self.assertEqual(len(rows_sha256([row])), 64)
        self.assertEqual(len(CANONICAL_SYSTEM_PROMPT_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
