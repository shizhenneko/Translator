# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.step2_translate import _strip_placeholder_backticks


def test_strip_placeholder_backticks_inline():
    text = "before `__CODE_BLOCK_001__` after"
    assert _strip_placeholder_backticks(text) == "before __CODE_BLOCK_001__ after"


def test_strip_placeholder_backticks_fenced_block():
    text = "```\n__CODE_BLOCK_001__\n```"
    assert _strip_placeholder_backticks(text) == "__CODE_BLOCK_001__"


def test_strip_placeholder_backticks_ignores_non_placeholder():
    text = "`code` and ```\ncode\n```"
    assert _strip_placeholder_backticks(text) == text
