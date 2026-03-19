from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.startup import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WINDOWS_VENV_DIR,
    WSL_VENV_DIR,
    build_serve_command,
    resolve_venv_dir,
)


def test_resolve_venv_dir_for_windows() -> None:
    assert resolve_venv_dir("Windows") == WINDOWS_VENV_DIR


def test_resolve_venv_dir_for_wsl() -> None:
    assert resolve_venv_dir("Linux") == WSL_VENV_DIR


def test_build_serve_command_defaults() -> None:
    assert build_serve_command("python") == [
        "python",
        "-m",
        "translator",
        "serve",
        "--host",
        DEFAULT_HOST,
        "--port",
        str(DEFAULT_PORT),
    ]
