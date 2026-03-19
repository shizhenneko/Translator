from __future__ import annotations

import os
import tempfile


def read_text(path: str) -> str:
    if not path:
        raise ValueError("input path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"input file not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"input path is not a file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def atomic_write_text(out_path: str, content: str) -> None:
    if not out_path:
        raise ValueError("output path is required")
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"output directory is not usable: {out_dir}")
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            _ = handle.write(content)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
