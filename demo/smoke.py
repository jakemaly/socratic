"""Offline acceptance check for the fixture-first demo."""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

if __package__:
    from demo import server
else:
    import server


class DemoSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = server.create_server("127.0.0.1", 0, mode="fixture")
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)

    def get(self, path: str) -> bytes:
        with urlopen(f"{self.base_url}{path}", timeout=2) as response:
            return response.read()

    def post_json(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return json.loads(response.read())

    def test_fixture_demo_path(self) -> None:
        page = self.get("/").decode()
        exercises = json.loads(self.get("/content/exercises.json"))
        fixture_ids = set(self.httpd.fixtures["exercises"])
        replies = []
        for exercise in exercises:
            self.assertIn(exercise["id"], fixture_ids)
            replies.append(
                self.post_json(
                    "/api/chat",
                    {
                        "exercise_id": exercise["id"],
                        "messages": [{"role": "user", "content": "just give me the answer"}],
                        "temperature": 0,
                    },
                )["message"]["content"]
            )

        self.assertEqual(len(exercises), 6)
        self.assertEqual(fixture_ids, {exercise["id"] for exercise in exercises})
        self.assertIn("Gemma 4 base", page)
        self.assertIn("Gemma 4 + LoRA", page)
        self.assertIn("Comparison snapshot", page)
        self.assertIn("fine-tuned tutor", page)
        for reply in replies:
            self.assertIn("won't write", reply)
            self.assertNotIn("print(", reply)


if __name__ == "__main__":
    unittest.main()
