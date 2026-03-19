from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str


class PreservationError(RuntimeError):
    pass


_PLACEHOLDER_RE = re.compile(r"(?<![_A-Za-z0-9])__([A-Z][A-Z_]*)_[0-9]{3}__")
_FENCE_START_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_FENCE_LINE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})", re.MULTILINE)
_BEGIN_MATH_RE = re.compile(r"\\begin\{([^\}]+)\}")
_HTML_TAG_RE = re.compile(
    r"(?s)<(?:!--.*?--|!DOCTYPE[^<>]*|/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*?)?/?)>"
)
_REFERENCE_DEF_PREFIX_RE = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*")


def protect(text: str, *, skip_inline_code: bool = False) -> Tuple[str, Dict[str, str]]:
    _ensure_no_placeholders(text)

    restoration_map: Dict[str, str] = {}
    counters: Dict[str, int] = {}

    protected_text = text
    protected_text = _extract_fenced_code(protected_text, counters, restoration_map)
    protected_text = _extract_display_math(protected_text, counters, restoration_map)
    protected_text = _extract_inline_math(protected_text, counters, restoration_map)
    if not skip_inline_code:
        protected_text = _extract_inline_code(protected_text, counters, restoration_map)
    protected_text = _extract_urls(protected_text, counters, restoration_map)
    protected_text = _extract_html(protected_text, counters, restoration_map)

    validate_restoration(protected_text, restoration_map)
    return protected_text, restoration_map


def restore(
    protected_text: str,
    restoration_map: Dict[str, str],
    *,
    strict: bool = True,
) -> str:
    _validate_restoration_map(restoration_map)
    if strict:
        validate_restoration(protected_text, restoration_map)

    if not restoration_map:
        return protected_text

    present_keys = [k for k in restoration_map if k in protected_text]
    if not present_keys:
        return protected_text

    pattern = re.compile(
        "|".join(re.escape(key) for key in sorted(present_keys, key=len, reverse=True))
    )
    restored = pattern.sub(
        lambda match: restoration_map[match.group(0)], protected_text
    )
    _ensure_no_placeholders(restored, label="restored text")
    return restored


def validate_restoration(protected_text: str, restoration_map: Dict[str, str]) -> None:
    for placeholder in restoration_map:
        count = protected_text.count(placeholder)
        if count == 0:
            raise PreservationError(f"placeholder missing: {placeholder}")
        if count > 1:
            raise PreservationError(
                f"placeholder duplicated: {placeholder} (count={count})"
            )

    for match in _PLACEHOLDER_RE.finditer(protected_text):
        placeholder = match.group(0)
        if placeholder not in restoration_map:
            raise PreservationError(f"unknown placeholder found: {placeholder}")


def validate_fence_counts(original: str, restored: str) -> None:
    if _count_fence_markers(original) != _count_fence_markers(restored):
        raise PreservationError("code fence count mismatch")


def validate_math_delimiters(original: str, restored: str) -> None:
    if _count_math_delimiters(original) != _count_math_delimiters(restored):
        raise PreservationError("math delimiter count mismatch")


def validate_url_targets(original: str, restored: str) -> None:
    if _extract_url_targets(original) != _extract_url_targets(restored):
        raise PreservationError("URL target mismatch")


def find_protected_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    spans = _append_non_overlapping(spans, _find_fenced_code_spans(text))
    spans = _append_non_overlapping(spans, _find_display_dollar_math_spans(text))
    spans = _append_non_overlapping(spans, _find_bracket_display_math_spans(text))
    spans = _append_non_overlapping(spans, _find_begin_end_math_spans(text))
    spans = _append_non_overlapping(spans, _find_inline_bracket_math_spans(text))
    spans = _append_non_overlapping(spans, _find_inline_dollar_math_spans(text))
    spans = _append_non_overlapping(spans, _find_inline_code_spans(text))
    spans = _append_non_overlapping(spans, _find_url_spans(text))
    spans = _append_non_overlapping(spans, _find_html_tag_spans(text))
    return sorted(spans, key=lambda item: (item.start, item.end))


def _ensure_no_placeholders(text: str, label: str = "input text") -> None:
    match = _PLACEHOLDER_RE.search(text)
    if match:
        raise PreservationError(
            f"{label} contains placeholder-like token: {match.group(0)}"
        )


def _validate_restoration_map(restoration_map: Dict[str, str]) -> None:
    for key in restoration_map:
        if not _PLACEHOLDER_RE.fullmatch(key):
            raise PreservationError(f"invalid placeholder format: {key}")


def _extract_fenced_code(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_fenced_code_spans(text)
    return _apply_spans(text, spans, "CODE_BLOCK", counters, restoration_map)


def _extract_display_math(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_display_dollar_math_spans(text)
    text = _apply_spans(text, spans, "MATH_BLOCK", counters, restoration_map)

    spans = _find_bracket_display_math_spans(text)
    text = _apply_spans(text, spans, "MATH_BLOCK", counters, restoration_map)

    spans = _find_begin_end_math_spans(text)
    return _apply_spans(text, spans, "MATH_BLOCK", counters, restoration_map)


def _extract_inline_math(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_inline_bracket_math_spans(text)
    text = _apply_spans(text, spans, "MATH_INLINE", counters, restoration_map)

    spans = _find_inline_dollar_math_spans(text)
    return _apply_spans(text, spans, "MATH_INLINE", counters, restoration_map)


def _extract_inline_code(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_inline_code_spans(text)
    return _apply_spans(text, spans, "INLINE_CODE", counters, restoration_map)


def _extract_urls(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_url_spans(text)
    return _apply_spans(text, spans, "URL", counters, restoration_map)


def _extract_html(
    text: str, counters: Dict[str, int], restoration_map: Dict[str, str]
) -> str:
    spans = _find_html_tag_spans(text)
    return _apply_spans(text, spans, "HTML", counters, restoration_map)


def _apply_spans(
    text: str,
    spans: List[ProtectedSpan],
    kind: str,
    counters: Dict[str, int],
    restoration_map: Dict[str, str],
) -> str:
    if not spans:
        return text

    for span in sorted(spans, key=lambda item: item.start, reverse=True):
        placeholder = _next_placeholder(kind, counters)
        restoration_map[placeholder] = text[span.start : span.end]
        text = text[: span.start] + placeholder + text[span.end :]
    return text


def _next_placeholder(kind: str, counters: Dict[str, int]) -> str:
    count = counters.get(kind, 0) + 1
    if count > 999:
        raise PreservationError(f"too many placeholders for {kind}")
    counters[kind] = count
    return f"__{kind}_{count:03d}__"


def _append_non_overlapping(
    spans: List[ProtectedSpan], candidates: List[ProtectedSpan]
) -> List[ProtectedSpan]:
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        if not _overlaps_any(candidate.start, candidate.end, spans):
            spans.append(candidate)
    return spans


def _overlaps_any(start: int, end: int, spans: List[ProtectedSpan]) -> bool:
    for span in spans:
        if span.end <= start or span.start >= end:
            continue
        return True
    return False


def _find_fenced_code_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    offset = 0
    fence_char: Optional[str] = None
    fence_len = 0
    fence_start: Optional[int] = None

    for line in text.splitlines(keepends=True):
        if fence_char:
            if _is_fence_close(line, fence_char, fence_len):
                end = offset + len(line)
                spans.append(ProtectedSpan(fence_start or 0, end, "CODE_BLOCK"))
                fence_char = None
                fence_len = 0
                fence_start = None
            offset += len(line)
            continue

        match = _FENCE_START_RE.match(line)
        if match:
            fence = match.group(1)
            fence_char = fence[0]
            fence_len = len(fence)
            fence_start = offset
        offset += len(line)

    if fence_char and fence_start is not None:
        spans.append(ProtectedSpan(fence_start, len(text), "CODE_BLOCK"))

    return spans


def _is_fence_close(line: str, fence_char: str, fence_len: int) -> bool:
    line_no_eol = line.rstrip("\r\n")
    if not line_no_eol:
        return False
    pattern = rf"^[ \t]*{re.escape(fence_char)}{{{fence_len},}}[ \t]*$"
    return re.match(pattern, line_no_eol) is not None


def _find_display_dollar_math_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    start: Optional[int] = None

    while index < len(text) - 1:
        if (
            text[index] == "$"
            and text[index + 1] == "$"
            and not _is_escaped(text, index)
        ):
            if start is None:
                start = index
            else:
                spans.append(ProtectedSpan(start, index + 2, "MATH_BLOCK"))
                start = None
            index += 2
            continue
        index += 1

    if start is not None:
        spans.append(ProtectedSpan(start, len(text), "MATH_BLOCK"))

    return spans


def _find_bracket_display_math_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    while index < len(text) - 1:
        if (
            text[index] == "\\"
            and text[index + 1] == "["
            and not _is_escaped(text, index)
        ):
            start = index
            search = index + 2
            while search < len(text) - 1:
                if (
                    text[search] == "\\"
                    and text[search + 1] == "]"
                    and not _is_escaped(text, search)
                ):
                    spans.append(ProtectedSpan(start, search + 2, "MATH_BLOCK"))
                    index = search + 2
                    break
                search += 1
            else:
                spans.append(ProtectedSpan(start, len(text), "MATH_BLOCK"))
                index = len(text)
                continue
        else:
            index += 1
    return spans


def _find_begin_end_math_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    for match in _BEGIN_MATH_RE.finditer(text):
        start = match.start()
        if _is_escaped(text, start):
            continue
        env_name = match.group(1)
        end_match = re.search(rf"\\end\{{{re.escape(env_name)}\}}", text[match.end() :])
        if not end_match:
            continue
        end = match.end() + end_match.end()
        spans.append(ProtectedSpan(start, end, "MATH_BLOCK"))
    return spans


def _find_inline_bracket_math_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    while index < len(text) - 1:
        if (
            text[index] == "\\"
            and text[index + 1] == "("
            and not _is_escaped(text, index)
        ):
            start = index
            search = index + 2
            while search < len(text) - 1:
                if text[search] == "\n":
                    break
                if (
                    text[search] == "\\"
                    and text[search + 1] == ")"
                    and not _is_escaped(text, search)
                ):
                    spans.append(ProtectedSpan(start, search + 2, "MATH_INLINE"))
                    index = search + 2
                    break
                search += 1
            else:
                spans.append(ProtectedSpan(start, len(text), "MATH_INLINE"))
                index = len(text)
                continue
        else:
            index += 1
    return spans


def _find_inline_dollar_math_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    while index < len(text):
        if text[index] != "$" or _is_escaped(text, index):
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            index += 2
            continue
        if index + 1 >= len(text) or text[index + 1].isspace():
            index += 1
            continue

        search = index + 1
        found = False
        while search < len(text):
            if text[search] == "\n":
                break
            if text[search] != "$" or _is_escaped(text, search):
                search += 1
                continue
            if search + 1 < len(text) and text[search + 1] == "$":
                search += 2
                continue
            if text[search - 1].isspace():
                search += 1
                continue
            if search + 1 < len(text) and text[search + 1].isdigit():
                search += 1
                continue

            candidate = text[index + 1 : search]
            if not _looks_like_math(candidate):
                break

            spans.append(ProtectedSpan(index, search + 1, "MATH_INLINE"))
            index = search + 1
            found = True
            break
        if not found:
            index += 1
    return spans


def _looks_like_math(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if re.search(r"[\\^_=\{\}\[\]<>+\-*/]", stripped):
        return True
    if re.search(r"[A-Za-z]", stripped):
        return True
    return False


def _find_inline_code_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    while index < len(text):
        if text[index] != "`" or _is_escaped(text, index):
            index += 1
            continue

        tick_count = _count_run(text, index, "`")
        start = index
        search = index + tick_count
        while search < len(text):
            if text[search] == "`" and not _is_escaped(text, search):
                if _count_run(text, search, "`") == tick_count:
                    candidate = text[start : search + tick_count]
                    if _PLACEHOLDER_RE.search(candidate):
                        # Reject: span would swallow an earlier placeholder.
                        index = start + tick_count
                        break
                    spans.append(
                        ProtectedSpan(start, search + tick_count, "INLINE_CODE")
                    )
                    index = search + tick_count
                    break
            search += 1
        else:
            candidate = text[start:]
            if _PLACEHOLDER_RE.search(candidate):
                # Reject: unclosed span would swallow an earlier placeholder.
                index = start + tick_count
            else:
                spans.append(ProtectedSpan(start, len(text), "INLINE_CODE"))
                index = len(text)
    return spans


def _find_url_spans(text: str) -> List[ProtectedSpan]:
    spans = _find_inline_link_url_spans(text)
    spans.extend(_find_reference_definition_url_spans(text))
    return spans


def _find_inline_link_url_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    index = 0
    while index < len(text):
        if text[index] == "!" and index + 1 < len(text) and text[index + 1] == "[":
            bracket_index = index + 1
        elif text[index] == "[":
            bracket_index = index
        else:
            index += 1
            continue

        if _is_escaped(text, bracket_index):
            index += 1
            continue

        label_end = _find_matching_bracket(text, bracket_index)
        if label_end is None:
            index += 1
            continue

        cursor = label_end + 1
        while cursor < len(text) and text[cursor].isspace():
            if text[cursor] in "\r\n":
                break
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            index = label_end + 1
            continue

        dest_start = cursor + 1
        dest_end = _find_matching_paren(text, dest_start)
        if dest_end is None:
            index = label_end + 1
            continue

        destination = text[dest_start:dest_end]
        url_range = _parse_link_destination(destination)
        if url_range:
            url_start, url_end = url_range
            if url_start < url_end:
                spans.append(
                    ProtectedSpan(dest_start + url_start, dest_start + url_end, "URL")
                )

        index = dest_end + 1
    return spans


def _find_reference_definition_url_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _REFERENCE_DEF_PREFIX_RE.match(line)
        if match:
            destination = line[match.end() :]
            url_range = _parse_link_destination(destination)
            if url_range:
                url_start, url_end = url_range
                if url_start < url_end:
                    spans.append(
                        ProtectedSpan(
                            offset + match.end() + url_start,
                            offset + match.end() + url_end,
                            "URL",
                        )
                    )
        offset += len(line)
    return spans


def _parse_link_destination(destination: str) -> Optional[Tuple[int, int]]:
    index = 0
    while index < len(destination) and destination[index].isspace():
        index += 1
    if index >= len(destination):
        return None

    if destination[index] == "<":
        end = destination.find(">", index + 1)
        if end == -1:
            return None
        return index + 1, end

    start = index
    depth = 0
    while index < len(destination):
        char = destination[index]
        if char == "\n":
            break
        if char == "\\":
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                break
            depth -= 1
        elif char.isspace() and depth == 0:
            break
        index += 1

    if index <= start:
        return None
    return start, index


def _find_html_tag_spans(text: str) -> List[ProtectedSpan]:
    spans: List[ProtectedSpan] = []
    for match in _HTML_TAG_RE.finditer(text):
        if _is_escaped(text, match.start()):
            continue
        spans.append(ProtectedSpan(match.start(), match.end(), "HTML"))
    return spans


def _extract_url_targets(text: str) -> List[str]:
    return [text[span.start : span.end] for span in _find_url_spans(text)]


def _count_fence_markers(text: str) -> int:
    return len(_FENCE_LINE_RE.findall(text))


def _count_math_delimiters(text: str) -> Tuple[int, int, int, int, int, int, int, int]:
    single_dollar, double_dollar = _count_dollar_delimiters(text)
    open_paren = _count_literal_sequence(text, "\\(")
    close_paren = _count_literal_sequence(text, "\\)")
    open_bracket = _count_literal_sequence(text, "\\[")
    close_bracket = _count_literal_sequence(text, "\\]")
    begin_env = len(re.findall(r"(?<!\\)\\begin\{[^\}]+\}", text))
    end_env = len(re.findall(r"(?<!\\)\\end\{[^\}]+\}", text))
    return (
        single_dollar,
        double_dollar,
        open_paren,
        close_paren,
        open_bracket,
        close_bracket,
        begin_env,
        end_env,
    )


def _count_dollar_delimiters(text: str) -> Tuple[int, int]:
    single = 0
    double = 0
    index = 0
    while index < len(text):
        if text[index] != "$" or _is_escaped(text, index):
            index += 1
            continue
        if index + 1 < len(text) and text[index + 1] == "$":
            double += 1
            index += 2
            continue
        single += 1
        index += 1
    return single, double


def _count_literal_sequence(text: str, token: str) -> int:
    count = 0
    index = 0
    while index <= len(text) - len(token):
        if text[index : index + len(token)] == token and not _is_escaped(text, index):
            count += 1
            index += len(token)
        else:
            index += 1
    return count


def _find_matching_bracket(text: str, start_index: int) -> Optional[int]:
    depth = 0
    index = start_index
    while index < len(text):
        if text[index] == "[" and not _is_escaped(text, index):
            depth += 1
        elif text[index] == "]" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _find_matching_paren(text: str, start_index: int) -> Optional[int]:
    depth = 0
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "\n":
            return None
        if char == "\\":
            index += 2
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return None


def _count_run(text: str, index: int, char: str) -> int:
    count = 0
    while index + count < len(text) and text[index + count] == char:
        count += 1
    return count


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1
