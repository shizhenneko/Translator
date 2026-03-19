from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.markdown_lint import MarkdownLintOptions, lint_markdown
from translator.markdown_normalize import normalize_markdown_for_preview
from translator.pipeline import enforce_markdown_guardrails
from translator.markdown_sanitize import sanitize_markdown_input


def _proj2_fixture() -> str:
    fixture = Path(__file__).parent / "fixtures" / "proj2_jina.md"
    return fixture.read_text(encoding="utf-8")


def test_normalize_proj2_removes_navigation_line():
    normalized = normalize_markdown_for_preview(
        sanitize_markdown_input(_proj2_fixture(), aggressive=True),
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "[Main](http://sp21.datastructur.es/index.html)" not in normalized


def test_normalize_proj2_removes_navigation_list_block():
    normalized = normalize_markdown_for_preview(
        sanitize_markdown_input(_proj2_fixture(), aggressive=True),
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "*   [A note on this spec]" not in normalized
    assert "*   [Overview of Gitlet]" not in normalized
    assert normalized.startswith("# Project 2: Gitlet | CS 61B Spring 2021\n## A note on this spec")


def test_normalize_proj2_deduplicates_title_and_collapses_inline_plaintext_fences():
    normalized = normalize_markdown_for_preview(
        sanitize_markdown_input(_proj2_fixture(), aggressive=True),
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "\nProject 2: Gitlet\n## A note on this spec\n" not in normalized
    assert "remote called\n```plaintext\nshared\n```" not in normalized
    assert "remote called `shared`, a repository called `repo`" in normalized


def test_normalize_repairs_literal_triple_backticks_plaintext_sequences():
    markdown = (
        "# Example\n\n"
        "A sentence ending with triple backticks (plaintext)\n"
        "gitlet.Main\n"
        "```\n"
        " and that it has a main method.\n"
    )
    normalized = normalize_markdown_for_preview(markdown)
    assert "triple backticks (plaintext)" not in normalized
    assert "`gitlet.Main` and that it has a main method." in normalized


def test_normalize_drops_plain_title_echo_after_h1():
    markdown = (
        "# Project 2: Gitlet | CS 61B Spring 2021\n\n"
        "Source: url https://example.com\n\n"
        "项目 2: Gitlet\n"
        "## 关于本规范的说明\n"
    )
    normalized = normalize_markdown_for_preview(
        markdown,
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "\n项目 2: Gitlet\n" not in normalized
    assert "## 关于本规范的说明" in normalized


def test_normalize_drops_leading_plain_title_echo_after_source_line():
    markdown = (
        "# Project 2: Gitlet | CS 61B Spring 2021\n\n"
        "Source: url https://example.com\n\n"
        "项目 2: Gitlet\n"
        "## 关于本说明的说明\n"
    )
    normalized = normalize_markdown_for_preview(
        markdown,
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "\n项目 2: Gitlet\n" not in normalized
    assert normalized.startswith("# Project 2: Gitlet | CS 61B Spring 2021\n\nSource: url https://example.com\n\n## 关于本说明的说明\n")


def test_normalize_drops_duplicate_h1_after_primary_title():
    markdown = (
        "# Project 2: Gitlet | CS 61B Spring 2021\n\n"
        "Source: url https://example.com\n\n"
        "# 项目 2：Gitlet | CS 61B Spring 2021\n\n"
        "## 关于本规范的说明\n"
    )
    normalized = normalize_markdown_for_preview(
        markdown,
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert normalized.count("# ") == 2  # one H1 plus one H2 marker
    assert "# 项目 2：Gitlet | CS 61B Spring 2021" not in normalized
    assert "## 关于本规范的说明" in normalized


def test_normalize_proj2_repairs_checkout_usage_lists():
    normalized = normalize_markdown_for_preview(
        sanitize_markdown_input(_proj2_fixture(), aggressive=True),
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "1.   ```plaintext" not in normalized
    assert "2.   ```plaintext" not in normalized
    assert "3.   ```plaintext" not in normalized
    assert "1. `java gitlet.Main checkout -- [file name]`" in normalized
    assert "2. `java gitlet.Main checkout [commit id] -- [file name]`" in normalized
    assert "3. `java gitlet.Main checkout [branch name]`" in normalized


def test_normalize_proj2_repairs_command_bullet_list_and_guardrails_pass():
    normalized = normalize_markdown_for_preview(
        sanitize_markdown_input(_proj2_fixture(), aggressive=True),
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "*   ```plaintext" not in normalized
    assert "* `init`" in normalized
    assert "* `checkout -- [file name]`" in normalized
    fixed = enforce_markdown_guardrails(normalized)
    issues = lint_markdown(
        fixed,
        MarkdownLintOptions(strict_renderer=True, max_safe_list_depth=1),
    )
    assert issues == []
