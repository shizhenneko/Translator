# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.markdown_sanitize import sanitize_markdown_input


def test_sanitize_repairs_empty_anchor_with_trailing_label():
    broken = "[](https://example.com/spec)Why specifications?\n"
    fixed = sanitize_markdown_input(broken)
    assert fixed == "[Why specifications?](https://example.com/spec)\n"


def test_sanitize_removes_exercise_residue_block():
    broken = (
        "Question line\n"
        "Yes (missing answer)\n"
        "No (missing answer)\n"
        "check explain\n"
        "\n"
        "## Keep Section\n"
        "Body\n"
    )
    fixed = sanitize_markdown_input(broken)
    assert "missing answer" not in fixed.lower()
    assert "check explain" not in fixed.lower()
    assert "## Keep Section" in fixed


def test_sanitize_splits_requires_effects_and_inline_fence_glue():
    broken = (
        "requires: `val` occurs exactly once in `arr`effects: returns index```ts\n"
        "function find() {}\n"
        "```\n"
    )
    fixed = sanitize_markdown_input(broken)
    assert "requires:" in fixed
    assert "\neffects:" in fixed
    assert "\n```ts\n" in fixed


def test_sanitize_is_idempotent():
    broken = (
        "[](https://example.com/spec)Why specifications?\n"
        "Yes (missing answer)\n"
        "check explain\n"
    )
    once = sanitize_markdown_input(broken)
    twice = sanitize_markdown_input(once)
    assert twice == once


def test_sanitize_normalizes_overlong_fence_opener_with_short_close():
    broken = "````ts\ncode\n```\n"
    fixed = sanitize_markdown_input(broken)
    assert fixed.startswith("```ts\n")
    assert fixed.endswith("```\n")


def test_sanitize_does_not_mutate_patterns_inside_fenced_block():
    broken = (
        "```md\n"
        "[](https://example.com/spec)Why specifications?\n"
        "Yes (missing answer)\n"
        "check explain\n"
        "requires: x effects: y```ts\n"
        "```\n"
    )
    fixed = sanitize_markdown_input(broken)
    assert fixed == broken


def test_sanitize_keeps_blank_lines_inside_fenced_block():
    broken = "```text\nline\n\n\n\nline2\n```\n"
    fixed = sanitize_markdown_input(broken)
    assert fixed == broken
