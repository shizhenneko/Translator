from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import List, Optional, Sequence, Set, Tuple

from .markdown_lint import (
    MarkdownLintOptions,
    RULE_LIST_FENCE_CHAINING,
    RULE_LIST_FENCE_COMPLEX_DEPTH,
    RULE_LIST_FENCE_INDENT,
    lint_markdown,
)


_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>(?:[-*+])|(?:\d{1,9}[.)]))[ \t]+"
)
_FENCE_START_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[^\n]*$")
_FENCE_ONLY_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$")
_FENCE_TOKEN_RE = re.compile(r"(`{3,}|~{3,})")
_TRIPLE_BACKTICKS_RE = re.compile(r"`{3,}(?P<lang>[A-Za-z0-9_-]+)?")
_PLAINTEXT_FENCE_RE = re.compile(r"^[`~]{3,}(?:plaintext|text)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class MarkdownAutofixOptions:
    strict_renderer: bool = True
    max_safe_list_depth: int = 1


def autofix_markdown(
    markdown: str, options: Optional[MarkdownAutofixOptions] = None
) -> str:
    resolved = _resolve_options(options)
    fixed = markdown
    fixed = _repair_inline_fence_sequences(fixed)
    fixed = _split_inline_list_fences(fixed)
    fixed = _normalize_list_fence_indentation(fixed)
    fixed = _replace_prose_triple_backticks(fixed)
    if resolved.strict_renderer:
        fixed = _stabilize_risky_list_fences(
            fixed, max_safe_list_depth=resolved.max_safe_list_depth
        )
        fixed = _normalize_plaintext_fence_blocks(fixed)
    return fixed


def normalize_list_fence_indentation(markdown: str) -> str:
    return _normalize_list_fence_indentation(markdown)


def _repair_inline_fence_sequences(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        if in_fence:
            fixed.append(line)
            close_match = _FENCE_ONLY_RE.match(line)
            if close_match:
                close = close_match.group("fence")
                if close and close[0] == fence_char and len(close) >= fence_len:
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
            continue

        split_line = _split_line_fence_glue(line)
        if split_line is not None:
            before, after = split_line
            fixed.append(before)
            line = after

        opening_match = _FENCE_START_RE.match(line)
        if opening_match:
            opening = opening_match.group("fence")
            if opening:
                in_fence = True
                fence_char = opening[0]
                fence_len = len(opening)
        fixed.append(line)

    joined = "\n".join(fixed)
    return _restore_terminal_newline(markdown, joined)


def _split_inline_list_fences(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        if in_fence:
            fixed.append(line)
            close_match = _FENCE_ONLY_RE.match(line)
            if close_match:
                close = close_match.group("fence")
                if close and close[0] == fence_char and len(close) >= fence_len:
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
            continue

        opening_match = _FENCE_START_RE.match(line)
        if opening_match:
            opening = opening_match.group("fence")
            if opening:
                in_fence = True
                fence_char = opening[0]
                fence_len = len(opening)
            fixed.append(line)
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if not list_match:
            fixed.append(line)
            continue

        inline_fence = _find_inline_fence(line, min_start=list_match.end())
        if inline_fence is None:
            fixed.append(line)
            continue

        before = line[: inline_fence.start()].rstrip()
        if not before.strip():
            fixed.append(line)
            continue

        remainder = line[inline_fence.start() :].strip()
        list_content_indent = " " * (_indent_width(list_match.group("indent")) + 4)
        fixed.append(before)
        fixed.append(list_content_indent + remainder)

        token = inline_fence.group(1)
        in_fence = True
        fence_char = token[0]
        fence_len = len(token)

    joined = "\n".join(fixed)
    return _restore_terminal_newline(markdown, joined)


def _normalize_list_fence_indentation(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    list_indent: Optional[int] = None
    list_active = False
    last_line_in_list = False

    index = 0
    while index < len(lines):
        line = lines[index]
        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            list_active = True
            last_line_in_list = True
            list_indent = _indent_width(list_match.group("indent")) + 4
            fixed.append(line)
            index += 1
            continue

        fence_match = _FENCE_START_RE.match(line)
        if (
            fence_match
            and list_active
            and last_line_in_list
            and list_indent is not None
        ):
            indent_spaces = " " * list_indent
            fence_indent = _indent_width(fence_match.group("indent"))
            fence = fence_match.group("fence")
            fence_len = len(fence)
            fence_char = fence[0]
            if fence_indent < list_indent:
                fixed.append(indent_spaces + line.lstrip())
            else:
                fixed.append(line)
            index += 1
            while index < len(lines):
                block_line = lines[index]
                prefix = block_line[: len(block_line) - len(block_line.lstrip(" \t"))]
                block_indent = _indent_width(prefix)
                if block_line.strip() and block_indent >= list_indent:
                    fixed.append(block_line)
                elif block_line.strip():
                    fixed.append(indent_spaces + block_line.lstrip(" \t"))
                else:
                    fixed.append(block_line if block_line.startswith(indent_spaces) else "")
                close_match = _FENCE_ONLY_RE.match(block_line)
                if close_match:
                    close_fence = close_match.group("fence")
                    if close_fence[0] == fence_char and len(close_fence) >= fence_len:
                        index += 1
                        break
                index += 1
            last_line_in_list = True
            continue

        if line.strip():
            if list_active and list_indent is not None:
                line_indent = _indent_width(line[: len(line) - len(line.lstrip(" \t"))])
                if line_indent < list_indent and not list_match:
                    list_active = False
                    list_indent = None
                    last_line_in_list = False
                else:
                    last_line_in_list = True
            else:
                last_line_in_list = False
        fixed.append(line)
        index += 1

    joined = "\n".join(fixed)
    return _restore_terminal_newline(markdown, joined)


def _replace_prose_triple_backticks(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0

    for line in lines:
        if in_fence:
            fixed.append(line)
            close_match = _FENCE_ONLY_RE.match(line)
            if close_match:
                close = close_match.group("fence")
                if close and close[0] == fence_char and len(close) >= fence_len:
                    in_fence = False
                    fence_char = ""
                    fence_len = 0
            continue

        fence_match = _FENCE_START_RE.match(line)
        if fence_match:
            opening = fence_match.group("fence")
            if opening:
                in_fence = True
                fence_char = opening[0]
                fence_len = len(opening)
            fixed.append(line)
            continue

        fixed.append(_replace_triple_backticks_in_line(line))

    joined = "\n".join(fixed)
    return _restore_terminal_newline(markdown, joined)


def _stabilize_risky_list_fences(markdown: str, max_safe_list_depth: int) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    lint_options = MarkdownLintOptions(
        strict_renderer=True, max_safe_list_depth=max(1, max_safe_list_depth)
    )
    risky_rules = {
        RULE_LIST_FENCE_INDENT,
        RULE_LIST_FENCE_COMPLEX_DEPTH,
        RULE_LIST_FENCE_CHAINING,
    }
    issues = lint_markdown(markdown, options=lint_options)
    risky_by_line = {}
    for issue in issues:
        if issue.rule_id not in risky_rules:
            continue
        risky_by_line.setdefault(issue.line, set()).add(issue.rule_id)
    risky_lines = sorted(risky_by_line.keys())
    if not risky_lines:
        return markdown

    processed_starts: Set[int] = set()
    for line_no in risky_lines:
        start = _seek_fence_open(lines, line_no - 1)
        if start is None or start in processed_starts:
            continue
        end = _seek_fence_close(lines, start)
        rules = risky_by_line.get(line_no, set())
        opening_match = _FENCE_START_RE.match(lines[start])
        if opening_match is None:
            continue
        source_indent = _indent_width(opening_match.group("indent"))
        target_indent = source_indent
        if RULE_LIST_FENCE_COMPLEX_DEPTH in rules:
            target_indent = min(target_indent, max_safe_list_depth * 4)
        if RULE_LIST_FENCE_CHAINING in rules:
            target_indent = 0
        _reindent_fence_block(lines, start, end, target_indent=target_indent)
        processed_starts.add(start)

    joined = "\n".join(lines)
    return _restore_terminal_newline(markdown, joined)


def _normalize_plaintext_fence_blocks(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    index = 0
    while index < len(lines):
        opening_match = _FENCE_START_RE.match(lines[index])
        if opening_match is None:
            index += 1
            continue
        without_indent = lines[index][len(opening_match.group("indent")) :].strip()
        if not _PLAINTEXT_FENCE_RE.match(without_indent):
            index += 1
            continue
        # Keep conventional list-content fences (4-space indent) unchanged.
        if _indent_width(opening_match.group("indent")) <= 4:
            index += 1
            continue
        end = _seek_fence_close(lines, index)
        _reindent_fence_block(lines, index, end, target_indent=4)
        index = end + 1

    joined = "\n".join(lines)
    return _restore_terminal_newline(markdown, joined)


def _seek_fence_open(lines: Sequence[str], index: int) -> Optional[int]:
    if index < 0 or index >= len(lines):
        return None
    if _FENCE_START_RE.match(lines[index]):
        return index
    for offset in range(1, 4):
        candidate = index + offset
        if candidate >= len(lines):
            break
        if _FENCE_START_RE.match(lines[candidate]):
            return candidate
    return None


def _seek_fence_close(lines: Sequence[str], start: int) -> int:
    opening_match = _FENCE_START_RE.match(lines[start])
    if opening_match is None:
        return start
    opening = opening_match.group("fence")
    fence_char = opening[0]
    fence_len = len(opening)
    for index in range(start + 1, len(lines)):
        close_match = _FENCE_ONLY_RE.match(lines[index])
        if close_match is None:
            continue
        close = close_match.group("fence")
        if close[0] == fence_char and len(close) >= fence_len:
            return index
    return len(lines) - 1


def _reindent_fence_block(
    lines: List[str], start: int, end: int, *, target_indent: int
) -> None:
    opening_match = _FENCE_START_RE.match(lines[start])
    if opening_match is None:
        return
    prefix = opening_match.group("indent")
    target_prefix = " " * max(0, target_indent)

    for index in range(start, end + 1):
        line = lines[index]
        if not line.strip():
            lines[index] = ""
            continue
        if line.startswith(prefix):
            stripped = line[len(prefix) :]
        else:
            stripped = line.lstrip(" \t")
        lines[index] = target_prefix + stripped


def _replace_triple_backticks_in_line(line: str) -> str:
    inline_mask = _inline_code_mask(line)
    cursor = 0
    out: List[str] = []
    changed = False

    for match in _TRIPLE_BACKTICKS_RE.finditer(line):
        start = match.start()
        end = match.end()
        if _is_masked(inline_mask, start, end):
            continue
        if not line[:start].strip():
            # Keep real fence markers that start at BOL.
            continue
        out.append(line[cursor:start])
        lang = match.group("lang")
        if lang:
            out.append(f"triple backticks ({lang})")
        else:
            out.append("triple backticks")
        cursor = end
        changed = True

    if not changed:
        return line

    out.append(line[cursor:])
    return "".join(out)


def _find_inline_fence(line: str, min_start: int) -> Optional[re.Match[str]]:
    inline_mask = _inline_code_mask(line)
    for match in _FENCE_TOKEN_RE.finditer(line):
        if match.start() < min_start:
            continue
        if _is_masked(inline_mask, match.start(), match.end()):
            continue
        if not line[: match.start()].strip():
            continue
        return match
    return None


def _split_line_fence_glue(line: str) -> Optional[Tuple[str, str]]:
    inline_mask = _inline_code_mask(line)
    for match in _FENCE_TOKEN_RE.finditer(line):
        start = match.start()
        if start <= 0:
            continue
        if not line[:start].strip():
            continue
        if line[start - 1].isspace():
            continue
        if _is_masked(inline_mask, start, match.end()):
            continue
        suffix = line[match.end() :]
        if re.match(r"[A-Za-z0-9_-]+", suffix) is None:
            continue
        before = line[:start].rstrip()
        after = line[start:].lstrip()
        if not before or not after:
            continue
        return before, after
    return None


def _indent_width(prefix: str) -> int:
    width = 0
    for char in prefix:
        width += 4 if char == "\t" else 1
    return width


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


def _is_masked(mask: List[bool], start: int, end: int) -> bool:
    return any(mask[pos] for pos in range(start, min(end, len(mask))))


def _restore_terminal_newline(original: str, updated: str) -> str:
    if original.endswith("\n") and not updated.endswith("\n"):
        return updated + "\n"
    if not original.endswith("\n") and updated.endswith("\n"):
        return updated.rstrip("\n")
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


def _resolve_options(
    options: Optional[MarkdownAutofixOptions],
) -> MarkdownAutofixOptions:
    if options is None:
        return MarkdownAutofixOptions(
            strict_renderer=_read_env_bool("TRANSLATOR_STRICT_RENDERER", True),
            max_safe_list_depth=_read_env_int(
                "TRANSLATOR_MAX_SAFE_LIST_DEPTH", default=1, minimum=1
            ),
        )
    return MarkdownAutofixOptions(
        strict_renderer=bool(options.strict_renderer),
        max_safe_list_depth=max(1, int(options.max_safe_list_depth)),
    )
