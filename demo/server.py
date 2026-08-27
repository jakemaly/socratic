"""Local server for the fixture-first Socratic Tutor demo."""
from __future__ import annotations

import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 32
MAX_MESSAGE_CHARS = 12_000
ALLOWED_STATIC_DIRS = ("demo", "content", "results")


class DemoHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying immutable demo configuration."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], fixtures: dict[str, Any], *, mode: str, model: Any = None):
        super().__init__(address, DemoRequestHandler)
        self.fixtures = fixtures
        self.mode = mode
        self.model = model
        self.static_root = ROOT.parent


def load_fixtures(path: Path = ROOT / "fixtures.json") -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("exercises"), dict):
        raise ValueError("fixtures must contain an exercises object")
    return data


def _live_model() -> Any:
    model_id = os.getenv("SOCRATIC_ADAPTER_MODEL")
    if not model_id:
        raise RuntimeError("SOCRATIC_ADAPTER_MODEL is required in live mode")
    try:
        from eval.client import OpenAICompatibleModel
    except ImportError as exc:  # pragma: no cover - only occurs outside the repo root
        raise RuntimeError("run the demo from the repository root") from exc
    return OpenAICompatibleModel(model_id)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    mode: str | None = None,
    fixtures_path: Path | None = None,
    model: Any | None = None,
) -> DemoHTTPServer:
    """Create a configured local server without starting its event loop."""
    fixtures = load_fixtures(fixtures_path or ROOT / "fixtures.json")
    effective_mode = (mode or os.getenv("SOCRATIC_DEMO_MODE", "fixture")).lower()
    if effective_mode not in {"fixture", "live"}:
        raise ValueError("mode must be fixture or live")
    if effective_mode == "live" and model is None:
        model = _live_model()
    return DemoHTTPServer((host, port), fixtures, mode=effective_mode, model=model)


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(encoded)


def _fixture_reply(exercise: dict[str, Any], messages: list[dict[str, str]]) -> str:
    latest_user = next(
        (
            message["content"].strip().casefold()
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        "",
    )
    replies = exercise["replies"]
    return replies.get(latest_user, replies["default"])


def _validate_chat_payload(payload: Any, exercises: dict[str, Any]) -> tuple[str, list[dict[str, str]], float]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    exercise_id = payload.get("exercise_id")
    if not isinstance(exercise_id, str) or exercise_id not in exercises:
        raise ValueError("unknown exercise_id")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if len(messages) > MAX_MESSAGES:
        raise ValueError("too many messages")

    clean_messages: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("message role is invalid")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        if len(content) > MAX_MESSAGE_CHARS:
            raise ValueError("message content is too long")
        clean_messages.append({"role": role, "content": content})

    temperature = payload.get("temperature", 0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")
    return exercise_id, clean_messages, float(temperature)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("demo: " + (format % args) + "\n")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path.startswith("/api/"):
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            _json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body is too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            exercise_id, messages, temperature = _validate_chat_payload(payload, self.server.fixtures["exercises"])
            if self.server.mode == "fixture":
                content = _fixture_reply(self.server.fixtures["exercises"][exercise_id], messages)
            else:
                if self.server.model is None:
                    raise RuntimeError("live adapter is not configured")
                from eval.judge import TUTOR_SYSTEM_PROMPT

                model_messages = messages
                if not messages or messages[0]["role"] != "system":
                    model_messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}, *messages]
                content = self.server.model.complete(model_messages, temperature=temperature, seed=0)
            _json_response(self, HTTPStatus.OK, {"message": {"role": "assistant", "content": content}})
        except ValueError as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - external adapter failures vary
            self.log_message("adapter request failed: %s", exc)
            _json_response(self, HTTPStatus.BAD_GATEWAY, {"error": "fine-tuned tutor unavailable"})

    def _serve_static(self, path: str) -> None:
        relative = unquote(path.lstrip("/")) or "demo/index.html"
        candidate = (self.server.static_root / relative).resolve()
        allowed_roots = [(self.server.static_root / name).resolve() for name in ALLOWED_STATIC_DIRS]
        if not any(root == candidate or root in candidate.parents for root in allowed_roots):
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if relative == "results/demo-evidence.json" and not candidate.is_file():
            _json_response(self, HTTPStatus.OK, {"available": False})
            return
        if not candidate.is_file():
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        body = candidate.read_bytes()
        if relative == "demo/index.html":
            body = body.replace(
                b'data-demo-mode="fixture"',
                f'data-demo-mode="{self.server.mode}"'.encode("ascii"),
            )
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="serve the Socratic Tutor demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("fixture", "live"), default=None)
    args = parser.parse_args()
    httpd = create_server(args.host, args.port, mode=args.mode)
    print(f"Socratic demo: http://{args.host}:{args.port} ({httpd.mode} mode)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
