#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
DETECTOR_SOURCE = ROOT / "tools" / "detect_faces.swift"
DETECTOR_BINARY = ROOT / ".local" / "detect_faces"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def ensure_detector() -> Path:
    DETECTOR_BINARY.parent.mkdir(exist_ok=True)
    needs_build = (
        not DETECTOR_BINARY.exists()
        or DETECTOR_BINARY.stat().st_mtime < DETECTOR_SOURCE.stat().st_mtime
    )
    if not needs_build:
        return DETECTOR_BINARY

    result = subprocess.run(
        ["swiftc", str(DETECTOR_SOURCE), "-o", str(DETECTOR_BINARY)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Swift detector build failed")
    return DETECTOR_BINARY


class LocalMaskingHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/detect-faces":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Missing image body"})
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Image is too large"})
            return

        suffix = ".png"
        content_type = self.headers.get("Content-Type", "")
        if "jpeg" in content_type or "jpg" in content_type:
            suffix = ".jpg"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=tempfile.gettempdir(),
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name
                temp_file.write(self.rfile.read(content_length))

            detector = ensure_detector()
            result = subprocess.run(
                [str(detector), temp_path],
                capture_output=True,
                text=True,
                cwd=ROOT,
                timeout=20,
            )
            if result.returncode != 0:
                self.send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": result.stderr.strip() or "Face detection failed"},
                )
                return

            payload = json.loads(result.stdout)
            self.send_json(HTTPStatus.OK, payload)
        except Exception as error:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    os.chdir(ROOT)
    port = int(os.environ.get("PORT", "5173"))
    server = ThreadingHTTPServer(("127.0.0.1", port), LocalMaskingHandler)
    print(f"Serving on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
