from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.markdown_autofix import MarkdownAutofixOptions, autofix_markdown
from translator.markdown_lint import MarkdownLintOptions, lint_markdown


def _rule_ids(
    markdown: str, options: Optional[MarkdownLintOptions] = None
) -> list[str]:
    return [issue.rule_id for issue in lint_markdown(markdown, options=options)]


def test_lint_flags_inline_fence_in_list_item_with_stable_rule_id():
    broken = (
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n"
    )
    issues = lint_markdown(broken)
    assert "FENCE_INLINE" in [issue.rule_id for issue in issues]
    inline_issue = next(issue for issue in issues if issue.rule_id == "FENCE_INLINE")
    assert inline_issue.line == 1


def test_lint_flags_literal_triple_backticks_in_prose():
    broken = "This note mentions ```plaintext in prose and should not open a fence.\n"
    issues = lint_markdown(broken)
    assert "PROSE_TRIPLE_BACKTICKS" in [issue.rule_id for issue in issues]
    prose_issue = next(
        issue for issue in issues if issue.rule_id == "PROSE_TRIPLE_BACKTICKS"
    )
    assert prose_issue.line == 1


def test_lint_flags_list_fence_indentation_issue():
    broken = "* **Usage**:\n\n```\ncmd\n```"
    issues = lint_markdown(broken)
    assert "LIST_FENCE_INDENT" in [issue.rule_id for issue in issues]


def test_lint_flags_deep_list_fence_in_strict_renderer_mode():
    broken = "* Parent\n    * Child\n        ```plaintext\n        cmd\n        ```\n"
    rule_ids = _rule_ids(
        broken,
        options=MarkdownLintOptions(strict_renderer=True, max_safe_list_depth=1),
    )
    assert "LIST_FENCE_COMPLEX_DEPTH" in rule_ids


def test_lint_can_disable_renderer_specific_rules():
    broken = "* Parent\n    * Child\n        ```plaintext\n        cmd\n        ```\n"
    rule_ids = _rule_ids(
        broken,
        options=MarkdownLintOptions(strict_renderer=False, max_safe_list_depth=1),
    )
    assert "LIST_FENCE_COMPLEX_DEPTH" not in rule_ids


def test_lint_flags_list_context_marker_drift():
    broken = "* First\n1. Second\n"
    rule_ids = _rule_ids(
        broken,
        options=MarkdownLintOptions(strict_renderer=True, max_safe_list_depth=1),
    )
    assert "LIST_CONTEXT_DRIFT" in rule_ids


def test_lint_flags_broken_empty_anchor_link():
    broken = "[](https://example.com/spec)Why specifications?\n"
    issues = lint_markdown(broken)
    assert "BROKEN_LINK_EMPTY_LABEL" in [issue.rule_id for issue in issues]


def test_lint_flags_interactive_residue():
    broken = "check explain\n"
    issues = lint_markdown(broken)
    assert "INTERACTIVE_RESIDUE" in [issue.rule_id for issue in issues]


def test_lint_flags_glued_fence_after_prose():
    broken = "effects: returns index```ts\n"
    issues = lint_markdown(broken)
    assert "FENCE_GLUE_AFTER_PROSE" in [issue.rule_id for issue in issues]


def test_autofix_splits_inline_fence_and_repairs_indent():
    broken = (
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n"
    )
    expected = (
        "* **Failure cases**: prints an error\n"
        "    ```plaintext\n"
        "    Found no commit with that message.\n"
        "    ```\n"
    )
    fixed = autofix_markdown(broken)
    assert fixed == expected
    assert _rule_ids(fixed) == []


def test_autofix_normalizes_list_fence_indentation():
    broken = "* **Usage**:\n\n```\ncmd\n```"
    expected = "* **Usage**:\n\n    ```\n    cmd\n    ```"
    fixed = autofix_markdown(broken)
    assert fixed == expected
    assert _rule_ids(fixed) == []


def test_autofix_flattens_deep_list_fence_for_renderer_stability():
    broken = "* Parent\n    * Child\n        ```plaintext\n        cmd\n        ```\n"
    fixed = autofix_markdown(
        broken,
        options=MarkdownAutofixOptions(
            strict_renderer=True,
            max_safe_list_depth=1,
        ),
    )
    assert "\n    ```plaintext\n" in fixed
    assert "        ```plaintext" not in fixed
    assert _rule_ids(
        fixed,
        options=MarkdownLintOptions(strict_renderer=True, max_safe_list_depth=1),
    ) == []


def test_autofix_rewrites_prose_triple_backticks():
    broken = "Do not place ```plaintext markers in prose.\n"
    expected = "Do not place triple backticks (plaintext) markers in prose.\n"
    fixed = autofix_markdown(broken)
    assert fixed == expected
    assert _rule_ids(fixed) == []


def test_autofix_repairs_glued_fence_before_prose_rewrite():
    broken = "effects: returns index```ts\nfunction find() {}\n```\n"
    fixed = autofix_markdown(broken)
    assert "```ts\n" in fixed
    assert "triple backticks (ts)" not in fixed
    assert "FENCE_GLUE_AFTER_PROSE" not in _rule_ids(fixed)


def test_autofix_is_idempotent():
    broken = (
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n"
        "Do not place ```plaintext markers in prose.\n"
    )
    once = autofix_markdown(broken)
    twice = autofix_markdown(once)
    assert twice == once


def test_autofix_indents_followup_text_after_list_fence():
    broken = (
        "* Parent\n"
        "    1. Failure case prints\n"
        "    ```plaintext\n"
        "    Nope.\n"
        "    ```\n"
        "  and exits.\n"
    )
    fixed = autofix_markdown(broken)
    assert "    and exits." in fixed
