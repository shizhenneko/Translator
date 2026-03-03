from __future__ import annotations

"""Markdown structure lint focused on VSCode markdown-it safety.

Rule IDs (stable contract):
- FENCE_UNBALANCED: opening fenced code block is not closed.
- FENCE_INLINE: ``` or ~~~ appears mid-line outside fenced blocks.
- LIST_FENCE_INDENT: fenced block under a list item is not indented as list content.
- LIST_FENCE_COMPLEX_DEPTH: fenced block nested under list depth that exceeds safe threshold.
- LIST_FENCE_CHAINING: chained fenced blocks under list context (rendering-risk pattern).
- LIST_CONTEXT_DRIFT: ambiguous list indentation or marker drift.
- BROKEN_LINK_EMPTY_LABEL: malformed [](...)Label link emitted by upstream source.
- FENCE_GLUE_AFTER_PROSE: glued prose + fenced block opener on the same line.
- INTERACTIVE_RESIDUE: interactive exercise residue line such as "check explain".
- PROSE_TRIPLE_BACKTICKS: literal ``` sequence appears in prose.
"""

from dataclasses import dataclass
import os
import re
from typing import List, Optional, Sequence


RULE_FENCE_UNBALANCED = "FENCE_UNBALANCED"
RULE_FENCE_INLINE = "FENCE_INLINE"
RULE_LIST_FENCE_INDENT = "LIST_FENCE_INDENT"
RULE_LIST_FENCE_COMPLEX_DEPTH = "LIST_FENCE_COMPLEX_DEPTH"
RULE_LIST_FENCE_CHAINING = "LIST_FENCE_CHAINING"
RULE_LIST_CONTEXT_DRIFT = "LIST_CONTEXT_DRIFT"
RULE_BROKEN_LINK_EMPTY_LABEL = "BROKEN_LINK_EMPTY_LABEL"
RULE_FENCE_GLUE_AFTER_PROSE = "FENCE_GLUE_AFTER_PROSE"
RULE_INTERACTIVE_RESIDUE = "INTERACTIVE_RESIDUE"
RULE_PROSE_TRIPLE_BACKTICKS = "PROSE_TRIPLE_BACKTICKS"

_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>(?:[-*+])|(?:\d{1,9}[.)]))[ \t]+"
)
_FENCE_START_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[^\n]*$")
_FENCE_ONLY_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$")
_FENCE_TOKEN_RE = re.compile(r"(`{3,}|~{3,})")
_BROKEN_LINK_EMPTY_LABEL_RE = re.compile(r"\[]\((?P<url>https?://[^)\s]+)\)(?=\S)")
_INTERACTIVE_RESIDUE_RE = re.compile(
    r"(?:check explain|\(missing answer\)|\(missing explanation\)|（缺失答案）|（缺失解释）)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MarkdownIssue:
    rule_id: str
    line: int
    message: str
    excerpt: str


@dataclass(frozen=True)
class MarkdownLintOptions:
    strict_renderer: bool = True
    max_safe_list_depth: int = 1


@dataclass(frozen=True)
class _ListContext:
    indent: int
    marker_kind: str


def lint_markdown(
    markdown: str, options: Optional[MarkdownLintOptions] = None
) -> List[MarkdownIssue]:
    resolved = _resolve_options(options)
    lines = markdown.splitlines()
    issues: List[MarkdownIssue] = []
    if not lines:
        return issues

    in_fence = False
    fence_char = ""
    fence_len = 0
    open_line = 0
    open_excerpt = ""
    current_fence_list_depth = 0
    last_nonblank_kind = "other"
    list_stack: List[_ListContext] = []

    for line_no, line in enumerate(lines, start=1):
        if in_fence:
            close_match = _FENCE_ONLY_RE.match(line)
            if close_match:
                close = close_match.group("fence")
                if close and close[0] == fence_char and len(close) >= fence_len:
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
                    open_line = 0
                    open_excerpt = ""
                    if line.strip():
                        if current_fence_list_depth > 0:
                            last_nonblank_kind = "fence_close_list"
                        else:
                            last_nonblank_kind = "fence_close"
                    current_fence_list_depth = 0
                    continue
            if line.strip():
                last_nonblank_kind = "fence_body"
            continue

        list_match = _LIST_ITEM_RE.match(line)
        fence_start_match = _FENCE_START_RE.match(line)
        if resolved.strict_renderer:
            if _BROKEN_LINK_EMPTY_LABEL_RE.search(line):
                issues.append(
                    MarkdownIssue(
                        rule_id=RULE_BROKEN_LINK_EMPTY_LABEL,
                        line=line_no,
                        message="malformed empty-label markdown link",
                        excerpt=_excerpt(line),
                    )
                )
            if _INTERACTIVE_RESIDUE_RE.search(line):
                issues.append(
                    MarkdownIssue(
                        rule_id=RULE_INTERACTIVE_RESIDUE,
                        line=line_no,
                        message="interactive exercise residue found",
                        excerpt=_excerpt(line),
                    )
                )

        if list_match:
            list_stack = _update_list_stack(
                list_stack=list_stack,
                list_match=list_match,
                issues=issues,
                line_no=line_no,
                line=line,
                options=resolved,
            )
        elif line.strip() and not fence_start_match:
            line_indent = _leading_indent_width(line)
            list_stack = _pop_out_of_list(list_stack, line_indent)
            if resolved.strict_renderer and list_stack:
                parent_indent = list_stack[-1].indent
                content_indent = parent_indent + 4
                if parent_indent < line_indent < content_indent:
                    issues.append(
                        MarkdownIssue(
                            rule_id=RULE_LIST_CONTEXT_DRIFT,
                            line=line_no,
                            message="ambiguous indentation inside list context",
                            excerpt=_excerpt(line),
                        )
                    )

        inline_mask = _inline_code_mask(line)
        for token_match in _FENCE_TOKEN_RE.finditer(line):
            start = token_match.start()
            end = token_match.end()
            if _is_masked(inline_mask, start, end):
                continue
            if not line[:start].strip():
                # Fence token at line-start position (valid fenced block marker).
                continue

            token = token_match.group(1)
            suffix = line[end:]
            if (
                resolved.strict_renderer
                and token.startswith("`")
                and re.match(r"[A-Za-z0-9_-]+", suffix) is not None
                and not line[start - 1].isspace()
            ):
                issues.append(
                    MarkdownIssue(
                        rule_id=RULE_FENCE_GLUE_AFTER_PROSE,
                        line=line_no,
                        message="fenced block opener is glued to prose",
                        excerpt=_excerpt(line),
                    )
                )
                break
            if token.startswith("`") and not list_match:
                issues.append(
                    MarkdownIssue(
                        rule_id=RULE_PROSE_TRIPLE_BACKTICKS,
                        line=line_no,
                        message="literal triple backticks found in prose",
                        excerpt=_excerpt(line),
                    )
                )
            else:
                issues.append(
                    MarkdownIssue(
                        rule_id=RULE_FENCE_INLINE,
                        line=line_no,
                        message="fence marker appears mid-line",
                        excerpt=_excerpt(line),
                    )
                )
            break

        if fence_start_match:
            opening = fence_start_match.group("fence")
            fence_indent = _indent_width(fence_start_match.group("indent"))
            list_context_active = bool(list_stack) and (
                last_nonblank_kind in {"list_item", "list_content", "fence_close_list"}
            )
            explicit_depth = _list_depth_for_indent(list_stack, fence_indent)
            list_depth = explicit_depth
            if list_depth == 0 and list_context_active:
                # The fence is likely intended as list content but mis-indented.
                list_depth = len(list_stack)

            if list_depth > 0:
                expected_source = (
                    list_stack[list_depth - 1]
                    if explicit_depth > 0
                    else list_stack[-1]
                )
                expected_indent = expected_source.indent + 4
                if fence_indent < expected_indent:
                    issues.append(
                        MarkdownIssue(
                            rule_id=RULE_LIST_FENCE_INDENT,
                            line=line_no,
                            message="fenced block under list item is not indented correctly",
                            excerpt=_excerpt(line),
                        )
                    )
                if (
                    resolved.strict_renderer
                    and list_depth > resolved.max_safe_list_depth
                ):
                    issues.append(
                        MarkdownIssue(
                            rule_id=RULE_LIST_FENCE_COMPLEX_DEPTH,
                            line=line_no,
                            message=(
                                "fenced block under deep list context "
                                f"(depth={list_depth}, max={resolved.max_safe_list_depth})"
                            ),
                            excerpt=_excerpt(line),
                        )
                    )
                if (
                    resolved.strict_renderer
                    and last_nonblank_kind == "fence_close_list"
                ):
                    issues.append(
                        MarkdownIssue(
                            rule_id=RULE_LIST_FENCE_CHAINING,
                            line=line_no,
                            message=(
                                "adjacent fenced blocks under list context are "
                                "rendering-risky"
                            ),
                            excerpt=_excerpt(line),
                        )
                    )
            if opening:
                in_fence = True
                fence_char = opening[0]
                fence_len = len(opening)
                open_line = line_no
                open_excerpt = _excerpt(line)
                current_fence_list_depth = list_depth
            if line.strip():
                if list_depth > 0:
                    last_nonblank_kind = "fence_open_list"
                else:
                    last_nonblank_kind = "fence_open"
            continue

        if line.strip():
            if list_match:
                last_nonblank_kind = "list_item"
            elif list_stack:
                last_nonblank_kind = "list_content"
            else:
                last_nonblank_kind = "other"

    if in_fence:
        issues.append(
            MarkdownIssue(
                rule_id=RULE_FENCE_UNBALANCED,
                line=open_line,
                message="opening fence is not closed",
                excerpt=open_excerpt,
            )
        )

    return issues


def format_issue_report(issues: Sequence[MarkdownIssue]) -> str:
    if not issues:
        return "No markdown lint issues."
    return "\n".join(
        f"{issue.rule_id} line {issue.line}: {issue.message} | {issue.excerpt}"
        for issue in issues
    )


def _indent_width(prefix: str) -> int:
    width = 0
    for char in prefix:
        width += 4 if char == "\t" else 1
    return width


def _excerpt(line: str) -> str:
    compact = line.strip()
    if len(compact) <= 120:
        return compact
    return compact[:117] + "..."


def _inline_code_mask(line: str) -> List[bool]:
    mask = [False] * len(line)
    open_start: Optional[int] = None
    open_len: Optional[int] = None
    index = 0
    while index < len(line):
        if line[index] != "`":
            index += 1
            continue
        end = index
        while end < len(line) and line[end] == "`":
            end += 1
        run_len = end - index
        if open_len is None:
            open_start = index
            open_len = run_len
        elif run_len == open_len and open_start is not None:
            for mark in range(open_start, end):
                mask[mark] = True
            open_start = None
            open_len = None
        index = end
    return mask


def _is_masked(mask: Sequence[bool], start: int, end: int) -> bool:
    if not mask:
        return False
    return any(mask[pos] for pos in range(start, min(end, len(mask))))


def _leading_indent_width(line: str) -> int:
    prefix = line[: len(line) - len(line.lstrip(" \t"))]
    return _indent_width(prefix)


def _list_depth_for_indent(list_stack: Sequence[_ListContext], indent: int) -> int:
    depth = 0
    for context in list_stack:
        if indent >= context.indent + 4:
            depth += 1
        else:
            break
    return depth


def _marker_kind(marker: str) -> str:
    if marker and marker[0].isdigit():
        return "ordered"
    return "bullet"


def _update_list_stack(
    *,
    list_stack: Sequence[_ListContext],
    list_match: re.Match[str],
    issues: List[MarkdownIssue],
    line_no: int,
    line: str,
    options: MarkdownLintOptions,
) -> List[_ListContext]:
    updated = list(list_stack)
    item_indent = _indent_width(list_match.group("indent"))
    marker_kind = _marker_kind(list_match.group("marker"))

    while updated and item_indent < updated[-1].indent:
        updated.pop()

    if updated and item_indent == updated[-1].indent:
        previous = updated[-1]
        if options.strict_renderer and previous.marker_kind != marker_kind:
            issues.append(
                MarkdownIssue(
                    rule_id=RULE_LIST_CONTEXT_DRIFT,
                    line=line_no,
                    message="list marker style changed at same indentation level",
                    excerpt=_excerpt(line),
                )
            )
        updated[-1] = _ListContext(indent=item_indent, marker_kind=marker_kind)
    elif not updated or item_indent > updated[-1].indent:
        updated.append(_ListContext(indent=item_indent, marker_kind=marker_kind))
    return updated


def _pop_out_of_list(
    list_stack: Sequence[_ListContext], line_indent: int
) -> List[_ListContext]:
    updated = list(list_stack)
    while updated and line_indent < updated[-1].indent + 4:
        updated.pop()
    return updated


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _read_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    if value < minimum:
        return minimum
    return value


def _resolve_options(options: Optional[MarkdownLintOptions]) -> MarkdownLintOptions:
    if options is None:
        return MarkdownLintOptions(
            strict_renderer=_read_env_bool("TRANSLATOR_STRICT_RENDERER", True),
            max_safe_list_depth=_read_env_int(
                "TRANSLATOR_MAX_SAFE_LIST_DEPTH", default=1, minimum=1
            ),
        )
    return MarkdownLintOptions(
        strict_renderer=bool(options.strict_renderer),
        max_safe_list_depth=max(1, int(options.max_safe_list_depth)),
    )
