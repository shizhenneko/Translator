from __future__ import annotations

from typing import List

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10001
WINDOWS_VENV_DIR = ".venv-windows"
WSL_VENV_DIR = ".venv"


def resolve_venv_dir(platform_name: str) -> str:
    normalized = platform_name.strip().lower()
    if normalized.startswith("win"):
        return WINDOWS_VENV_DIR
    return WSL_VENV_DIR


def build_serve_command(
    python_executable: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> List[str]:
    if not python_executable:
        raise ValueError("python executable is required")
    if not host:
        raise ValueError("host is required")
    if port <= 0:
        raise ValueError("port must be positive")
    return [
        python_executable,
        "-m",
        "translator",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
