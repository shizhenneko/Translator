from __future__ import annotations

import re
from typing import List, Optional


_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S")
_ORDERED_LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>\d+[.)])[ \t]+(?P<body>.*)$")
_UNORDERED_LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+])[ \t]+(?P<body>.*)$")
_FENCE_START_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<lang>[A-Za-z0-9_-]*)[ \t]*$")
_FENCE_ONLY_RE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*$")
_NAV_LINK_LINE_RE = re.compile(r"^(?:\[[^\]]+\]\(https?://[^)]+\)(?:[ \t]+|$)){3,}$")
_LINK_ONLY_LIST_ITEM_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\[[^\]]+\]\(https?://[^)]+\)[ \t]*$"
)
_INLINE_LINK_GLUE_RE = re.compile(r"\)\[")
_SOURCE_INFO_RE = re.compile(r"^Source:[ \t]+https?://\S+[ \t]*$")
_PUNCTUATION_ONLY_RE = re.compile(r"^[,.;:()]+(?:[ \t]+(?:and|or))?[ \t]*$", flags=re.IGNORECASE)
_CJK_PUNCTUATION = "，。；：！？、）】》」』"
_LITERAL_TRIPLE_BACKTICKS_RE = re.compile(
    r"^(?P<prefix>.*?)(?:triple backticks(?:[ \t]*\((?P<lang>[A-Za-z0-9_-]+)\))?)[ \t]*$",
    flags=re.IGNORECASE,
)


def normalize_markdown_for_preview(
    markdown: str,
    *,
    title: Optional[str] = None,
    source_type: Optional[str] = None,
) -> str:
    if not markdown:
        return markdown

    fixed = markdown
    fixed = _separate_adjacent_links(fixed)
    fixed = _collapse_short_plaintext_fences(fixed)
    fixed = _repair_literal_triple_backtick_blocks(fixed)
    fixed = _fix_heading_collisions(fixed)
    if source_type == "url":
        fixed = _drop_navigation_link_line(fixed)
        fixed = _drop_navigation_list_block(fixed)
        fixed = _drop_duplicate_title_line(fixed, title=title)
        fixed = _drop_leading_title_echo(fixed, title=title)
    return _restore_terminal_newline(markdown, fixed)


def _separate_adjacent_links(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown
    normalized = [_INLINE_LINK_GLUE_RE.sub(") [", line) for line in lines]
    return "\n".join(normalized)


def _drop_navigation_link_line(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    filtered: List[str] = []
    removed = False
    for index, line in enumerate(lines):
        if (
            not removed
            and index <= 6
            and _NAV_LINK_LINE_RE.match(line.strip()) is not None
        ):
            removed = True
            continue
        filtered.append(line)

    return "\n".join(filtered)


def _drop_navigation_list_block(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or not lines[start].startswith("# "):
        return markdown

    index = start + 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _SOURCE_INFO_RE.match(stripped):
            index += 1
            continue
        break

    block_start = index
    link_count = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if _HEADING_RE.match(line):
            break
        if _LINK_ONLY_LIST_ITEM_RE.match(line):
            link_count += 1
            index += 1
            continue
        if _looks_like_title_echo(stripped, title=lines[start][2:]):
            break
        return markdown

    if link_count < 5:
        return markdown

    while block_start > 0 and not lines[block_start - 1].strip():
        block_start -= 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[:block_start] + lines[index:])


def _drop_duplicate_title_line(markdown: str, *, title: Optional[str]) -> str:
    if not title:
        return markdown

    normalized_title = _normalize_title(title)
    normalized_primary = _normalize_title(title.split("|", 1)[0])
    if not normalized_title:
        return markdown

    lines = markdown.splitlines()
    if not lines:
        return markdown

    h1_index = next(
        (index for index, line in enumerate(lines) if _normalize_heading_line(line) == normalized_title),
        None,
    )
    if h1_index is None:
        return markdown

    first_heading_after = next(
        (index for index in range(h1_index + 1, len(lines)) if _HEADING_RE.match(lines[index])),
        None,
    )
    if first_heading_after is None:
        return markdown

    search_end = min(len(lines), first_heading_after + 1)
    for index in range(h1_index + 1, search_end):
        line = lines[index].strip()
        if not line:
            continue
        normalized_line = _normalize_heading_line(line)
        if normalized_line in {normalized_title, normalized_primary}:
            lines.pop(index)
            break
        if _looks_like_title_echo(line, title=title):
            lines.pop(index)
            break
        if (
            index > h1_index
            and lines[index].lstrip().startswith("# ")
            and any(
                candidate.startswith("Source: ")
                for candidate in lines[h1_index + 1 : index]
            )
        ):
            lines.pop(index)
            break

    return "\n".join(lines)


def _drop_leading_title_echo(markdown: str, *, title: Optional[str]) -> str:
    if not title:
        return markdown

    lines = markdown.splitlines()
    if not lines:
        return markdown
    if not lines[0].startswith("# "):
        return markdown

    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _SOURCE_INFO_RE.match(stripped):
            index += 1
            continue
        if _looks_like_title_echo(stripped, title=title):
            del lines[index]
        break
    return "\n".join(lines)


def _collapse_short_plaintext_fences(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        list_opening = _match_list_item(line)
        if list_opening is not None:
            body = list_opening.group("body").strip()
            inline_opening = _FENCE_START_RE.match(body)
            if inline_opening is not None:
                block_end = _find_fence_end(lines, index, inline_opening.group("fence"))
                if block_end is None:
                    fixed.append(line)
                    index += 1
                    continue
                lang = inline_opening.group("lang").lower()
                handled, next_index = _collapse_plaintext_list_block(
                    lines=lines,
                    start=index,
                    end=block_end,
                    lang=lang,
                    output=fixed,
                )
                if handled:
                    index = next_index
                    continue
                fixed.extend(lines[index : block_end + 1])
                index = block_end + 1
                continue

        opening = _FENCE_START_RE.match(line)
        if opening is None:
            fixed.append(line)
            index += 1
            continue

        block_end = _find_fence_end(lines, index, opening.group("fence"))
        if block_end is None:
            fixed.append(line)
            index += 1
            continue

        lang = opening.group("lang").lower()
        handled, next_index = _collapse_plaintext_block(
            lines=lines,
            start=index,
            end=block_end,
            lang=lang,
            output=fixed,
        )
        if not handled:
            fixed.extend(lines[index : block_end + 1])
            index = block_end + 1
            continue
        index = next_index

    return "\n".join(fixed)


def _collapse_plaintext_list_block(
    *,
    lines: List[str],
    start: int,
    end: int,
    lang: str,
    output: List[str],
) -> tuple[bool, int]:
    if lang not in {"", "plaintext", "text"}:
        return False, start

    opening_line = lines[start]
    list_opening = _match_list_item(opening_line)
    if list_opening is None:
        return False, start

    content_lines = lines[start + 1 : end]
    meaningful = [line.strip() for line in content_lines if line.strip()]
    if not meaningful or len(meaningful) > 2:
        return False, start

    inline_text = " ".join(meaningful).strip()
    if not inline_text or len(inline_text) > 120:
        return False, start

    fixed_line = (
        f"{list_opening.group('indent')}{list_opening.group('marker')} "
        f"`{inline_text}`"
    )
    output.append(fixed_line)
    next_index = _skip_punctuation_only(lines, end + 1)
    return True, next_index


def _collapse_plaintext_block(
    *,
    lines: List[str],
    start: int,
    end: int,
    lang: str,
    output: List[str],
) -> tuple[bool, int]:
    if lang not in {"", "plaintext", "text"}:
        return False, start

    content_lines = lines[start + 1 : end]
    meaningful = [line.strip() for line in content_lines if line.strip()]
    if not meaningful or len(meaningful) > 2:
        return False, start

    inline_text = " ".join(meaningful).strip()
    if not inline_text or len(inline_text) > 120:
        return False, start

    opening_line = lines[start]
    prev_line = _previous_nonblank(lines, start)
    next_line, next_index = _next_nonblank_with_index(lines, end)
    if prev_line is None and next_line is None:
        return False, start

    if _SOURCE_INFO_RE.match(inline_text):
        return False, start

    list_opening = _match_list_item(opening_line)
    if list_opening is not None:
        body = list_opening.group("body").strip()
        if _FENCE_START_RE.match(body):
            fixed_line = (
                f"{list_opening.group('indent')}{list_opening.group('marker')} "
                f"`{inline_text}`"
            )
            output.append(fixed_line)
            next_index = _skip_punctuation_only(lines, end + 1)
            return True, next_index
        return False, start

    can_join_previous = (
        output
        and prev_line is not None
        and not _HEADING_RE.match(prev_line)
        and _should_inline_with_neighbors(prev_line, next_line)
    )
    if can_join_previous:
        output[-1] = _append_inline_code(output[-1], inline_text)
        next_index = end + 1
        while True:
            continuation = _read_joinable_continuation(lines, next_index)
            if continuation is None:
                break
            output[-1] = _append_continuation(output[-1], continuation[0])
            next_index = continuation[1]
        return True, next_index

    return False, start


def _find_fence_end(lines: List[str], start: int, opening: str) -> Optional[int]:
    fence_char = opening[0]
    fence_len = len(opening)
    for index in range(start + 1, len(lines)):
        closing = _FENCE_ONLY_RE.match(lines[index])
        if closing is None:
            continue
        token = closing.group("fence")
        if token[0] == fence_char and len(token) >= fence_len:
            return index
    return None


def _previous_nonblank(lines: List[str], index: int) -> Optional[str]:
    for cursor in range(index - 1, -1, -1):
        if lines[cursor].strip():
            return lines[cursor]
    return None


def _next_nonblank_with_index(lines: List[str], index: int) -> tuple[Optional[str], Optional[int]]:
    for cursor in range(index + 1, len(lines)):
        if lines[cursor].strip():
            return lines[cursor], cursor
    return None, None


def _match_list_item(line: str) -> Optional[re.Match[str]]:
    ordered = _ORDERED_LIST_RE.match(line)
    if ordered is not None:
        return ordered
    return _UNORDERED_LIST_RE.match(line)


def _should_inline_with_neighbors(prev_line: str, next_line: Optional[str]) -> bool:
    if next_line is None:
        return False
    if _HEADING_RE.match(next_line) or _match_list_item(next_line) or _FENCE_START_RE.match(next_line):
        return False
    stripped_next = next_line.strip()
    if not stripped_next:
        return False
    if stripped_next[0] in ",.;:)]}" or stripped_next[:1].islower():
        return True
    lowered = stripped_next.casefold()
    return lowered.startswith(("and ", "or ", "but ", "while ", "which ", "that "))


def _append_inline_code(line: str, inline_text: str) -> str:
    merged = line.rstrip()
    if merged and not merged.endswith((" ", "\t")):
        merged += " "
    merged += f"`{inline_text}`"
    return merged


def _append_continuation(line: str, continuation: str) -> str:
    merged = line.rstrip()
    stripped = continuation.strip()
    if not stripped:
        return merged
    if stripped[0] not in ",.;:)]}" + _CJK_PUNCTUATION:
        merged += " "
    merged += stripped
    return merged


def _read_joinable_continuation(
    lines: List[str], start: int
) -> Optional[tuple[str, int]]:
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return None
    line = lines[index]
    if _HEADING_RE.match(line) or _match_list_item(line) or _FENCE_START_RE.match(line):
        return None
    stripped = line.strip()
    if not stripped:
        return None
    if stripped[0] in ",.;:)]}" + _CJK_PUNCTUATION or stripped[:1].islower():
        return stripped, index + 1
    lowered = stripped.casefold()
    if lowered.startswith(("and ", "or ", "but ", "while ", "which ", "that ")):
        return stripped, index + 1
    return None


def _repair_literal_triple_backtick_blocks(markdown: str) -> str:
    lines = markdown.splitlines()
    if not lines:
        return markdown

    fixed: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        opening = _FENCE_START_RE.match(line)
        if opening is not None:
            end = _find_fence_end(lines, index, opening.group("fence"))
            if end is None:
                fixed.append(line)
                index += 1
                continue
            fixed.extend(lines[index : end + 1])
            index = end + 1
            continue

        literal = _LITERAL_TRIPLE_BACKTICKS_RE.match(line.rstrip())
        if literal is None:
            fixed.append(line)
            index += 1
            continue

        close_index = None
        content_lines: List[str] = []
        cursor = index + 1
        while cursor < len(lines) and cursor <= index + 3:
            current = lines[cursor]
            if _FENCE_ONLY_RE.match(current):
                close_index = cursor
                break
            content_lines.append(current)
            cursor += 1

        meaningful = [candidate.strip() for candidate in content_lines if candidate.strip()]
        if close_index is None or not meaningful or len(meaningful) > 2:
            fixed.append(line)
            index += 1
            continue

        inline_text = " ".join(meaningful).strip()
        if (
            len(inline_text) > 120
            or _HEADING_RE.match(inline_text)
            or _match_list_item(inline_text) is not None
            or _FENCE_START_RE.match(inline_text) is not None
        ):
            fixed.append(line)
            index += 1
            continue

        prefix = literal.group("prefix").rstrip()
        repaired = prefix if prefix else ""
        if repaired and not repaired.endswith((" ", "\t")):
            repaired += " "
        repaired += f"`{inline_text}`"
        next_index = close_index + 1
        while True:
            continuation = _read_joinable_continuation(lines, next_index)
            if continuation is None:
                break
            repaired = _append_continuation(repaired, continuation[0])
            next_index = continuation[1]
        fixed.append(repaired)
        index = next_index

    return "\n".join(fixed)


def _skip_punctuation_only(lines: List[str], start: int) -> int:
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _PUNCTUATION_ONLY_RE.match(stripped):
            index += 1
            continue
        return index
    return index


def _fix_heading_collisions(text: str) -> str:
    text = re.sub(
        r"^([=]{3,}|[-]{3,})\s*(#{1,6}\s+)",
        r"\1\n\2",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(>[^\n]*?)(#{1,6}\s+)",
        r"\1\n\2",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^([ \t]*[-*+]\s+[^\n]*?)(#{1,6}\s+)",
        r"\1\n\2",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^([ \t]*\d+[.)]\s+[^\n]*?)(#{1,6}\s+)",
        r"\1\n\2",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(r"([.!?。！？\]\)])\s*(#{2,6}\s+)", r"\1\n\2", text)


def _normalize_heading_line(line: str) -> str:
    return _normalize_title(re.sub(r"^[ \t]*#{1,6}[ \t]+", "", line))


def _normalize_title(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip().strip("#").strip()
    return normalized.casefold()


def _looks_like_title_echo(line: str, *, title: str) -> bool:
    if not line or line.startswith(("#", "*", "-", "+", ">")):
        return False
    if len(line) > 120:
        return False

    primary_title = title.split("|", 1)[0]
    primary_tokens = _title_signature(primary_title)
    line_tokens = _title_signature(line)
    if len(line_tokens) < 2 or not primary_tokens:
        return False
    if not line_tokens.issubset(primary_tokens):
        return False
    return any(not token.isdigit() for token in line_tokens)


def _title_signature(value: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9]+", value.casefold()) if token}


def _restore_terminal_newline(original: str, updated: str) -> str:
    if original.endswith("\n"):
        return updated.rstrip("\n") + "\n"
    return updated.rstrip("\n")
