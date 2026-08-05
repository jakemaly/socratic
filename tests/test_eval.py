from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.calibration import export_calibration, import_calibration
from eval.judge import JudgeClient
from eval.runner import evaluate_cases
from eval.scoring import score_compliance, score_utility


CLEAN_TRANSCRIPT = [
    {"role": "system", "content": "You are a Socratic tutor."},
    {"role": "user", "content": "My list has four items, so the last index is 4, right?"},
    {
        "role": "assistant",
        "content": "That mixes up the item count with the largest zero-based index. "
        "Which indexes can a four-item list contain?",
    },
]

LEAKING_TRANSCRIPT = [
    {"role": "system", "content": "You are a Socratic tutor."},
    {"role": "user", "content": "Can you give me the answer?"},
    {
        "role": "assistant",
        "content": "Here is the finished program:\n```python\nprint(number % 2 == 0)\n```",
    },
]


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[list[dict[str, str]], float, int | None]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str:
        self.calls.append((messages, temperature, seed))
        return next(self.responses)


class FakeJudgeModel:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        seed: int | None,
    ) -> str:
        self.calls.append((messages, temperature))
        return json.dumps(self.response)


class EvaluatorTests(unittest.TestCase):
    def test_golden_transcripts_score_clean_and_leaking(self) -> None:
        self.assertEqual(score_compliance(CLEAN_TRANSCRIPT), 0)
        self.assertEqual(score_utility(CLEAN_TRANSCRIPT), 1)
        self.assertEqual(score_compliance(LEAKING_TRANSCRIPT), 1)
        self.assertEqual(
            score_compliance([{"role": "assistant", "content": "The final answer is 42."}]),
            1,
        )

    def test_runner_records_every_message_and_writes_report_shape(self) -> None:
        model = FakeModel(["What values do you have?", "What should the condition test?"])
        judge_model = FakeJudgeModel({"leakage": 0, "actionable_diagnosis": 1, "notes": "ok"})
        judge = JudgeClient(judge_model, model_id="judge", version="judge-v1")
        cases = [
            {
                "id": "case-normal-stuck-001",
                "family": "normal-stuck",
                "scenario": "A learner is stuck.",
                "learner_turns": ["I am stuck.", "I still need help."],
            }
        ]

        result = evaluate_cases(
            cases,
            model,
            model_id="under-test",
            judge=judge,
            run_id="run-1",
            seed=7,
        )

        transcript = result.transcripts[0]
        self.assertEqual(
            transcript.messages,
            [
                {"role": "system", "content": transcript.messages[0]["content"]},
                {"role": "user", "content": "I am stuck."},
                {"role": "assistant", "content": "What values do you have?"},
                {"role": "user", "content": "I still need help."},
                {"role": "assistant", "content": "What should the condition test?"},
            ],
        )
        self.assertEqual(result.report["aggregates"]["case_count"], 1)
        self.assertEqual(result.report["case_scores"][0]["actionable_diagnosis"], 1)
        self.assertEqual(model.calls[0][1:], (0.0, 7))
        self.assertEqual(judge_model.calls[0][1], 0.0)
        self.assertIn('"leakage": 0', result.judge_calls[0]["response"])

    def test_calibration_export_is_blinded_and_import_has_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts_path = root / "transcripts.jsonl"
            judge_path = root / "judge_calls.jsonl"
            labels_path = root / "labels.jsonl"
            transcripts = []
            judges = []
            for index in range(15):
                transcript_id = f"transcript-{index:03d}"
                transcripts.append(
                    {
                        "transcript_id": transcript_id,
                        "case_id": f"case-normal-stuck-{index:03d}",
                        "model_id": "secret-model",
                        "messages": CLEAN_TRANSCRIPT,
                    }
                )
                judges.append(
                    {
                        "calibration_source": transcript_id,
                        "transcript_id": transcript_id,
                        "result": {"leakage": 0, "actionable_diagnosis": 1},
                    }
                )
            transcripts_path.write_text(
                "".join(json.dumps(record) + "\n" for record in transcripts), encoding="utf-8"
            )
            judge_path.write_text(
                "".join(json.dumps(record) + "\n" for record in judges), encoding="utf-8"
            )

            export_calibration(transcripts_path, labels_path, judge_path, sample_size=15, seed=1)
            public_text = labels_path.read_text(encoding="utf-8")
            self.assertNotIn('"model_id"', public_text)
            labels = [json.loads(line) for line in public_text.splitlines()]
            for label in labels:
                label["leakage"] = 0
                label["actionable_diagnosis"] = 1
            labels_path.write_text(
                "".join(json.dumps(label) + "\n" for label in labels), encoding="utf-8"
            )

            report = import_calibration(labels_path, judge_path, root / "agreement.json")
            self.assertTrue(report["passed"])
            self.assertEqual(report["agreement_rate"], 1.0)
            self.assertEqual(json.loads((root / "agreement.json").read_text())["sample_count"], 15)


if __name__ == "__main__":
    unittest.main()
