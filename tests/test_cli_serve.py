from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import translator.cli as cli


def test_serve_command_dispatches_to_web_app(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_serve_app(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("translator.web.app.serve_app", fake_serve_app)

    exit_code = cli.main(
        [
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "12000",
            "--job-dir",
            "tmp-jobs",
            "--workers",
            "4",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "host": "0.0.0.0",
        "port": 12000,
        "job_base_dir": "tmp-jobs",
        "max_workers": 4,
    }
