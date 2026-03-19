from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_python_module_launcher_help() -> None:
    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "translator", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "translate" in result.stdout
