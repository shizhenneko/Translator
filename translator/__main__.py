from __future__ import annotations

import os
import sys


def _ensure_src_root_on_path() -> None:
    src_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "src")
    )
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def main() -> int:
    _ensure_src_root_on_path()
    from translator.cli import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
