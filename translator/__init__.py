"""Compatibility package for repo-root execution.

Implementation modules live in `src/translator/`. This package only exposes
that implementation so `python -m translator` and `import translator.*` both
work from the repository root without duplicating runtime code.
"""

from __future__ import annotations

import os

_SRC_PACKAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "src", "translator")
)

__path__ = [_SRC_PACKAGE_DIR]
