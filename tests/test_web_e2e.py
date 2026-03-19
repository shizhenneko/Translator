from __future__ import annotations

import io
import socket
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest
import requests
from werkzeug.serving import make_server

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.web.app import create_app


class _ServerThread(threading.Thread):
    def __init__(self, app, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)

    def run(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_batch_job_http_flow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_translate_document(**kwargs: object) -> str:
        source_value = str(kwargs["source_value"])
        out_path = str(kwargs["out_path"])
        write_text = kwargs["write_text"]
        if source_value.endswith("/fail"):
            raise RuntimeError("boom")
        write_text(out_path, f"# {source_value}\n")
        return f"# {source_value}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    app = create_app(job_base_dir=str(tmp_path / "jobs"), max_workers=1)
    port = _find_free_port()
    server = _ServerThread(app, "127.0.0.1", port)
    server.start()

    try:
        response = requests.get(f"http://127.0.0.1:{port}/", timeout=10)
        assert response.status_code == 200

        create = requests.post(
            f"http://127.0.0.1:{port}/api/jobs/url-file",
            files={
                "file": (
                    "urls.txt",
                    b"https://example.com/a\nhttps://example.com/fail\n",
                    "text/plain",
                )
            },
            data={"concurrency": "1"},
            timeout=10,
        )
        assert create.status_code == 202
        payload = create.json()
        job_id = payload["job_id"]

        deadline = time.time() + 10
        job = payload
        while time.time() < deadline:
            polled = requests.get(
                f"http://127.0.0.1:{port}/api/jobs/{job_id}",
                timeout=10,
            )
            polled.raise_for_status()
            job = polled.json()
            if job["status"] in {"success", "failed"}:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("job did not finish in time")

        assert job["status"] == "success"
        assert job["successful_count"] == 1
        assert job["failed_count"] == 1
        assert job["download_url"] == f"/api/jobs/{job_id}/download"

        download = requests.get(
            f"http://127.0.0.1:{port}{job['download_url']}",
            timeout=10,
        )
        download.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            names = archive.namelist()
            assert len(names) == 1
            assert names[0].endswith(".md")
            assert archive.read(names[0]).decode("utf-8") == "# https://example.com/a\n"
    finally:
        app.extensions["translation_job_manager"].close()
        server.shutdown()
        server.join(timeout=5)
