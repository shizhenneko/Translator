from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_serve_command_starts_http_server(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent
    port = _find_free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "translator",
            "serve",
            "--port",
            str(port),
            "--job-dir",
            str(tmp_path / "jobs"),
            "--workers",
            "1",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                response = requests.get(f"http://127.0.0.1:{port}/", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.1)
        else:
            stdout, stderr = process.communicate(timeout=5)
            raise AssertionError(
                f"serve command did not start in time\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )

        assert "网页转中文 Markdown" in response.text
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.skipif(
    not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MOONSHOT_API_KEY")),
    reason="DEEPSEEK_API_KEY not set - skipping real translate command test",
)
def test_translate_command_with_real_api(tmp_path: Path) -> None:
    repo_root = Path(__file__).parent.parent
    input_path = tmp_path / "input.md"
    input_path.write_text("# Sample\n\nHello world.\n", encoding="utf-8")
    output_path = tmp_path / "output.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "translator",
            "translate",
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--concurrency",
            "1",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Source: file" in content
