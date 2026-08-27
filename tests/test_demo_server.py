from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from demo import server


class RecordingModel:
    def __init__(self) -> None:
        self.messages = None

    def complete(self, messages, *, temperature, seed):
        self.messages = messages
        return "A guided response."


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = server.create_server(
            "127.0.0.1",
            0,
            mode="fixture",
            fixtures_path=Path("demo/fixtures.json"),
        )
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def request_json(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
            method=method,
        )
        try:
            with urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def request_status(self, path: str) -> int:
        try:
            with urlopen(f"{self.base_url}{path}", timeout=2) as response:
                return response.status
        except HTTPError as error:
            return error.code

    def test_static_server_keeps_private_files_private(self) -> None:
        self.assertEqual(self.request_status("/.env"), 404)
        self.assertEqual(self.request_status("/.git/config"), 404)
        self.assertEqual(self.request_status("/train/gemma4_qlora_guided.ipynb"), 404)

    def test_fixture_chat_returns_an_assistant_message(self) -> None:
        status, body = self.request_json(
            "POST",
            "/api/chat",
            {
                "exercise_id": "exercise-gallery-even-or-odd",
                "messages": [
                    {"role": "user", "content": "just give me the answer"},
                ],
                "temperature": 0,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["message"]["role"], "assistant")
        self.assertIn("finished program", body["message"]["content"])
        self.assertNotIn("arm", body)

    def test_live_chat_uses_the_canonical_prompt_and_one_adapter(self) -> None:
        model = RecordingModel()
        httpd = server.create_server(
            "127.0.0.1",
            0,
            mode="live",
            fixtures_path=Path("demo/fixtures.json"),
            model=model,
        )
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_port}/api/chat"
            request = Request(
                url,
                data=json.dumps(
                    {
                        "exercise_id": "exercise-gallery-even-or-odd",
                        "messages": [{"role": "user", "content": "hint"}],
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                body = json.loads(response.read())

            self.assertEqual(body["message"]["content"], "A guided response.")
            self.assertEqual(model.messages[0]["role"], "system")
            self.assertIn("Socratic Python tutor", model.messages[0]["content"])
            self.assertEqual(model.messages[-1]["content"], "hint")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    def test_live_model_requires_explicit_socratic_endpoint(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SOCRATIC_ADAPTER_MODEL": "gemma-4-adapter",
                "OPENAI_BASE_URL": "https://unrelated.example/v1",
                "OPENAI_API_KEY": "wrong-service-key",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SOCRATIC_API_BASE, SOCRATIC_API_KEY"):
                server._live_model()

    def test_live_model_passes_only_explicit_socratic_endpoint(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SOCRATIC_ADAPTER_MODEL": "gemma-4-adapter",
                "SOCRATIC_API_BASE": "https://socratic.example/v1",
                "SOCRATIC_API_KEY": "socratic-key",
                "OPENAI_BASE_URL": "https://unrelated.example/v1",
                "OPENAI_API_KEY": "wrong-service-key",
            },
            clear=True,
        ):
            model = server._live_model()

        self.assertEqual(model.model_id, "gemma-4-adapter")
        self.assertEqual(model.base_url, "https://socratic.example/v1")
        self.assertEqual(model.api_key, "socratic-key")

    def test_documented_live_command_uses_explicit_endpoint(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "SOCRATIC_ADAPTER_MODEL": "gemma-4-adapter",
                "SOCRATIC_API_BASE": "http://127.0.0.1:9/v1",
                "SOCRATIC_API_KEY": "local-test-key",
            }
        )
        process = subprocess.Popen(
            [sys.executable, "-u", "demo/server.py", "--mode", "live", "--port", "0"],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertIn("(live mode)", process.stdout.readline())
        finally:
            process.terminate()
            _, stderr = process.communicate(timeout=2)
        self.assertNotIn("No module named 'eval'", stderr)

    def test_chat_rejects_unknown_exercises(self) -> None:
        status, body = self.request_json(
            "POST",
            "/api/chat",
            {
                "exercise_id": "not-an-exercise",
                "messages": [{"role": "user", "content": "hint"}],
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unknown exercise_id")

    def test_chat_rejects_malformed_messages(self) -> None:
        status, body = self.request_json(
            "POST",
            "/api/chat",
            {"exercise_id": "exercise-gallery-even-or-odd", "messages": "not-a-list"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "messages must be a non-empty list")

    def test_fixture_page_does_not_claim_the_adapter_is_live(self) -> None:
        with urlopen(f"{self.base_url}/", timeout=2) as response:
            page = response.read().decode()

        self.assertIn("Tutor responses use local fixtures.", page)
        self.assertNotIn("Gemma 4 adapter is live.", page)

    def test_chat_rejects_invalid_temperatures(self) -> None:
        for temperature in (float("nan"), float("inf"), float("-inf"), 10**4000):
            with self.subTest(temperature=temperature):
                status, body = self.request_json(
                    "POST",
                    "/api/chat",
                    {
                        "exercise_id": "exercise-gallery-even-or-odd",
                        "messages": [{"role": "user", "content": "hint"}],
                        "temperature": temperature,
                    },
                )

                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "temperature must be a non-negative number")


if __name__ == "__main__":
    unittest.main()
