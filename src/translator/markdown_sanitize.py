from __future__ import annotations

import re
from typing import List, Optional


_EMPTY_ANCHOR_WITH_LABEL_RE = re.compile(
    r"\[]\((?P<url>https?://[^)\s]+)\)(?P<label>\S[^\n]*)"
)
_EMPTY_ANCHOR_RE = re.compile(r"\[]\((?P<url>https?://[^)\s]+)\)")
_EXERCISE_ANCHOR_RE = re.compile(r"#@ex-[A-Za-z0-9_-]+", flags=re.IGNORECASE)
_RESIDUE_LINE_RE = re.compile(
    r"(?:check explain|\(missing answer\)|\(missing explanation\)|（缺失答案）|（缺失解释）)",
    flags=re.IGNORECASE,
)
_REQUIRES_EFFECTS_RE = re.compile(
    r"(?P<prefix>.*\brequires:\s*)(?P<requires>.*?)(?P<effects>\beffects:\s*.+)",
    flags=re.IGNORECASE,
)
_FENCE_GLUE_RE = re.compile(r"(?P<fence>`{3,}|~{3,})(?P<lang>[A-Za-z0-9_-]+)")
_FENCE_START_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*$")
_OVERLONG_FENCE_WITH_LANG_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{4,}|~{4,})(?P<lang>[A-Za-z0-9_-]+)[ \t]*$"
)
_DOUBLE_FENCE_RE = re.compile(r"^(`{6,}|~{6,})$", flags=re.MULTILINE)
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S")
_SETEXT_RE = re.compile(r"^[=-]{3,}[ \t]*$")
_HR_RE = re.compile(r"^[ \t]{0,3}(?:[-*_][ \t]*){3,}$")


def sanitize_markdown_input(markdown: str, *, aggressive: bool = True) -> str:
    if not markdown:
        return markdown

    normalized = _normalize_double_fence_lines(markdown)
    lines = normalized.splitlines()
    normalized_lines = _normalize_non_fence_lines(lines)
    if aggressive:
        normalized_lines = _drop_exercise_residue_blocks(normalized_lines)
    normalized_lines = _expand_glued_requires_effects(normalized_lines)
    normalized_lines = _repair_inline_fence_glue(normalized_lines)
    normalized_lines = _normalize_overlong_fence_openers(normalized_lines)
    normalized_lines = _collapse_excess_blank_lines(normalized_lines)
    updated = "\n".join(normalized_lines)
    return _restore_terminal_newline(markdown, updated)


def _normalize_double_fence_lines(markdown: str) -> str:
    return _DOUBLE_FENCE_RE.sub(
        lambda m: (
            m.group(1)[: len(m.group(1)) // 2]
            + "\n"
            + m.group(1)[: len(m.group(1)) // 2]
        ),
        markdown,
    )


def _normalize_line(line: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group("url")
        label = match.group("label").strip()
        if not label:
            return ""
        # If label is already a markdown link, keep it and drop the empty-anchor prefix.
        if label.startswith("["):
            return label
        return f"[{label}]({url})"

    updated = _EMPTY_ANCHOR_WITH_LABEL_RE.sub(repl, line)
    updated = _EMPTY_ANCHOR_RE.sub("", updated)
    # Adjacent links without separator are hard to read in preview.
    updated = re.sub(r"\)\[", ") [", updated)
    return updated.rstrip()


def _normalize_non_fence_lines(lines: List[str]) -> List[str]:
    normalized: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in lines:
        if in_fence:
            normalized.append(line)
            close = _fence_close_token(line, fence_char=fence_char, min_len=fence_len)
            if close:
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue

        normalized_line = _normalize_line(line)
        normalized.append(normalized_line)

        opening = _fence_open_token(normalized_line)
        if opening:
            in_fence = True
            fence_char = opening[0]
            fence_len = len(opening)
    return normalized


def _drop_exercise_residue_blocks(lines: List[str]) -> List[str]:
    if not lines:
        return lines
    in_fence = _fence_state_mask(lines)
    drop = [False] * len(lines)
    for index, line in enumerate(lines):
        if in_fence[index]:
            continue
        if not _is_check_explain(line):
            if _is_residue_line(line) or _EXERCISE_ANCHOR_RE.search(line):
                drop[index] = True
            continue
        start = _find_exercise_start(lines, index, in_fence=in_fence)
        end = index
        for pos in range(start, end + 1):
            drop[pos] = True
        # Remove a single trailing blank line after the removed block.
        if end + 1 < len(lines) and not lines[end + 1].strip():
            drop[end + 1] = True

    return [line for index, line in enumerate(lines) if not drop[index]]


def _find_exercise_start(
    lines: List[str], check_explain_index: int, *, in_fence: List[bool]
) -> int:
    window_start = max(0, check_explain_index - 120)
    trigger = check_explain_index
    for index in range(check_explain_index, window_start - 1, -1):
        if in_fence[index]:
            break
        line = lines[index]
        if _EXERCISE_ANCHOR_RE.search(line) or _is_residue_line(line):
            trigger = index
            continue
        if line.strip() == "":
            break

    return trigger


def _expand_glued_requires_effects(lines: List[str]) -> List[str]:
    expanded: List[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in lines:
        if in_fence:
            expanded.append(line)
            close = _fence_close_token(line, fence_char=fence_char, min_len=fence_len)
            if close:
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue

        match = _REQUIRES_EFFECTS_RE.match(line)
        if match is None:
            expanded.append(line)
            opening = _fence_open_token(line)
            if opening:
                in_fence = True
                fence_char = opening[0]
                fence_len = len(opening)
            continue
        prefix = match.group("prefix").rstrip()
        requires = match.group("requires").strip()
        effects = match.group("effects").strip()
        expanded.append(f"{prefix}{requires}".rstrip())
        expanded.append(effects)
        opening = _fence_open_token(effects)
        if opening:
            in_fence = True
            fence_char = opening[0]
            fence_len = len(opening)
    return expanded


def _repair_inline_fence_glue(lines: List[str]) -> List[str]:
    repaired: List[str] = []
    in_fence = False
    for line in lines:
        if in_fence:
            repaired.append(line)
            if _is_fence_only_line(line):
                in_fence = False
            continue

        if _FENCE_START_RE.match(line):
            repaired.append(line)
            in_fence = True
            continue

        split_line = _split_glued_fence_line(line)
        if split_line is None:
            repaired.append(line)
            continue
        repaired.extend(split_line)
        if split_line and _FENCE_START_RE.match(split_line[-1]):
            in_fence = True
    return repaired


def _split_glued_fence_line(line: str) -> Optional[List[str]]:
    inline_mask = _inline_code_mask(line)
    for match in _FENCE_GLUE_RE.finditer(line):
        start = match.start("fence")
        if start <= 0:
            continue
        if not line[:start].strip():
            continue
        if line[start - 1].isspace():
            continue
        if _is_masked(inline_mask, start, match.end("lang")):
            continue
        before = line[:start].rstrip()
        after = line[start:].lstrip()
        if not before or not after:
            continue
        return [before, after]
    return None


def _collapse_excess_blank_lines(lines: List[str]) -> List[str]:
    collapsed: List[str] = []
    blank_run = 0
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line in lines:
        if in_fence:
            collapsed.append(line)
            close = _fence_close_token(line, fence_char=fence_char, min_len=fence_len)
            if close:
                in_fence = False
                fence_char = ""
                fence_len = 0
                blank_run = 0
            continue

        opening = _fence_open_token(line)
        if opening:
            collapsed.append(line)
            in_fence = True
            fence_char = opening[0]
            fence_len = len(opening)
            blank_run = 0
            continue

        if not line.strip():
            blank_run += 1
            if blank_run <= 2:
                collapsed.append("")
            continue
        blank_run = 0
        collapsed.append(line)
    return collapsed


def _normalize_overlong_fence_openers(lines: List[str]) -> List[str]:
    fixed = list(lines)
    for index, line in enumerate(fixed):
        match = _OVERLONG_FENCE_WITH_LANG_RE.match(line)
        if match is None:
            continue
        fence = match.group("fence")
        fence_char = fence[0]
        fence_len = len(fence)
        if _has_fence_close(fixed, index + 1, fence_char, min_len=fence_len):
            continue
        if not _has_fence_close(fixed, index + 1, fence_char, min_len=3):
            continue
        indent = match.group("indent")
        lang = match.group("lang")
        fixed[index] = f"{indent}{fence_char * 3}{lang}"
    return fixed


def _fence_state_mask(lines: List[str]) -> List[bool]:
    states = [False] * len(lines)
    in_fence = False
    fence_char = ""
    fence_len = 0
    for index, line in enumerate(lines):
        states[index] = in_fence
        if in_fence:
            close = _fence_close_token(line, fence_char=fence_char, min_len=fence_len)
            if close:
                in_fence = False
                fence_char = ""
                fence_len = 0
            continue
        opening = _fence_open_token(line)
        if opening:
            in_fence = True
            fence_char = opening[0]
            fence_len = len(opening)
    return states


def _fence_open_token(line: str) -> Optional[str]:
    match = _FENCE_START_RE.match(line)
    if match is None:
        return None
    token_match = re.match(r"[ \t]*(`{3,}|~{3,})", line)
    if token_match is None:
        return None
    return token_match.group(1)


def _fence_close_token(line: str, *, fence_char: str, min_len: int) -> Optional[str]:
    close_match = re.match(r"^[ \t]*(`{3,}|~{3,})[ \t]*$", line)
    if close_match is None:
        return None
    close = close_match.group(1)
    if not close:
        return None
    if close[0] != fence_char:
        return None
    if len(close) < min_len:
        return None
    return close


def _has_fence_close(
    lines: List[str], start_index: int, fence_char: str, min_len: int
) -> bool:
    pattern = re.compile(rf"^[ \t]*{re.escape(fence_char)}{{{min_len},}}[ \t]*$")
    for index in range(start_index, len(lines)):
        if pattern.match(lines[index]):
            return True
    return False


def _is_section_boundary(line: str) -> bool:
    if _HEADING_RE.match(line):
        return True
    if _SETEXT_RE.match(line):
        return True
    return _HR_RE.match(line) is not None


def _is_check_explain(line: str) -> bool:
    return line.strip().lower() == "check explain"


def _is_residue_line(line: str) -> bool:
    return _RESIDUE_LINE_RE.search(line) is not None


def _is_fence_only_line(line: str) -> bool:
    return re.match(r"^[ \t]*(`{3,}|~{3,})[ \t]*$", line) is not None


def _inline_code_mask(line: str) -> List[bool]:
    mask = [False] * len(line)
    open_start = None
    open_len = None
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
            for pos in range(open_start, end):
                mask[pos] = True
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
