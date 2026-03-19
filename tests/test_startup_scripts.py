from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_start_wsl_script_help() -> None:
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        ["bash", "./start_wsl.sh", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
    assert "translator web console" in result.stdout


@pytest.mark.skipif(
    shutil.which("powershell.exe") is None,
    reason="powershell.exe not available",
)
def test_start_windows_script_help() -> None:
    repo_root = Path(__file__).parent.parent
    windows_script_path = subprocess.run(
        ["wslpath", "-w", str(repo_root / "start_windows.ps1")],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            windows_script_path,
            "-Help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Usage:" in result.stdout
