from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .preservation import ProtectedSpan, find_protected_spans


@dataclass(frozen=True)
class ChunkPlanEntry:
    chunk_id: str
    source_text: str
    separators: List[str]


@dataclass(frozen=True)
class _Segment:
    text: str
    separator: str


@dataclass(frozen=True)
class _Section:
    text: str
    start: int


@dataclass(frozen=True)
class _ChunkDraft:
    source_text: str
    separators: List[str]


_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+")
_BLANK_LINE_RE = re.compile(r"(?:\r?\n[ \t]*){2,}")


def build_chunk_plan(text: str, max_chunk_chars: int) -> List[ChunkPlanEntry]:
    if max_chunk_chars <= 0:
        raise ValueError("max-chunk-chars must be positive")
    if not text:
        return []

    protected_spans = find_protected_spans(text)
    sections = _split_by_headings(text, protected_spans)
    drafts: List[_ChunkDraft] = []

    for section in sections:
        segments = _split_section_segments(
            section.text, section.start, protected_spans, max_chunk_chars
        )
        drafts.extend(_pack_segments(segments, max_chunk_chars))

    return _assign_chunk_ids(drafts)


def reconstruct_from_chunks(chunks: Sequence[ChunkPlanEntry]) -> str:
    return "".join(chunk.source_text for chunk in chunks)


def chunk_plan_payload(chunks: Sequence[ChunkPlanEntry]) -> List[Dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_text": chunk.source_text,
            "separators": chunk.separators,
        }
        for chunk in chunks
    ]


def _split_by_headings(
    text: str, protected_spans: Sequence[ProtectedSpan]
) -> List[_Section]:
    boundaries = [0]
    offset = 0
    for line in text.splitlines(keepends=True):
        if _HEADING_RE.match(line) and not _index_in_spans(offset, protected_spans):
            if offset != boundaries[-1]:
                boundaries.append(offset)
        offset += len(line)
    if boundaries[-1] != len(text):
        boundaries.append(len(text))

    sections: List[_Section] = []
    for start, end in zip(boundaries, boundaries[1:]):
        sections.append(_Section(text=text[start:end], start=start))
    return sections


def _split_section_segments(
    section_text: str,
    section_start: int,
    protected_spans: Sequence[ProtectedSpan],
    max_chunk_chars: int,
) -> List[_Segment]:
    parts: List[_Segment] = []
    last_index = 0

    for match in _BLANK_LINE_RE.finditer(section_text):
        sep_start = match.start()
        sep_end = match.end()
        abs_start = section_start + sep_start
        abs_end = section_start + sep_end
        if _overlaps_spans(abs_start, abs_end, protected_spans):
            continue
        text_part = section_text[last_index:sep_start]
        separator = section_text[sep_start:sep_end]
        parts.extend(_expand_part(text_part, separator, max_chunk_chars))
        last_index = sep_end

    tail = section_text[last_index:]
    parts.extend(_expand_part(tail, "", max_chunk_chars))
    return parts


def _expand_part(text: str, separator: str, max_chunk_chars: int) -> List[_Segment]:
    if not text and not separator:
        return []
    text_len = len(text)
    sep_len = len(separator)
    if text_len > max_chunk_chars:
        return _force_split(text, separator, max_chunk_chars)
    if text_len + sep_len <= max_chunk_chars:
        return [_Segment(text=text, separator=separator)]
    if sep_len > max_chunk_chars:
        raise ValueError("separator exceeds max-chunk-chars")
    return [_Segment(text=text, separator=""), _Segment(text="", separator=separator)]


_SENTENCE_END_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _force_split(text: str, separator: str, max_chunk_chars: int) -> List[_Segment]:
    segments: List[_Segment] = []
    remaining = text

    while len(remaining) > max_chunk_chars:
        cut = max_chunk_chars
        best = -1
        for match in _SENTENCE_END_RE.finditer(remaining, 0, cut):
            best = match.end()
        if best <= 0:
            best = remaining.rfind("\n", 0, cut)
        if best <= 0:
            best = remaining.rfind(" ", 0, cut)
        if best <= 0:
            best = cut
        segments.append(_Segment(text=remaining[:best], separator=""))
        remaining = remaining[best:]

    segments.append(_Segment(text=remaining, separator=separator))
    return segments


def _pack_segments(
    segments: Iterable[_Segment], max_chunk_chars: int
) -> List[_ChunkDraft]:
    drafts: List[_ChunkDraft] = []
    current_text_parts: List[str] = []
    current_separators: List[str] = []
    current_len = 0

    for segment in segments:
        seg_len = len(segment.text) + len(segment.separator)
        if seg_len > max_chunk_chars:
            raise ValueError("segment exceeds max-chunk-chars")
        if current_len + seg_len > max_chunk_chars and current_len > 0:
            drafts.append(
                _ChunkDraft(
                    source_text="".join(current_text_parts),
                    separators=list(current_separators),
                )
            )
            current_text_parts = []
            current_separators = []
            current_len = 0

        current_text_parts.append(segment.text)
        current_text_parts.append(segment.separator)
        current_separators.append(segment.separator)
        current_len += seg_len

    if current_text_parts:
        drafts.append(
            _ChunkDraft(
                source_text="".join(current_text_parts),
                separators=list(current_separators),
            )
        )

    return drafts


def _assign_chunk_ids(drafts: Sequence[_ChunkDraft]) -> List[ChunkPlanEntry]:
    if not drafts:
        return []
    width = max(4, len(str(len(drafts))))
    chunks: List[ChunkPlanEntry] = []
    for index, draft in enumerate(drafts, start=1):
        chunk_id = f"chunk-{index:0{width}d}"
        chunks.append(
            ChunkPlanEntry(
                chunk_id=chunk_id,
                source_text=draft.source_text,
                separators=draft.separators,
            )
        )
    return chunks


def _index_in_spans(index: int, spans: Sequence[ProtectedSpan]) -> bool:
    return _span_containing(index, spans) is not None


def _overlaps_spans(start: int, end: int, spans: Sequence[ProtectedSpan]) -> bool:
    for span in spans:
        if span.end <= start or span.start >= end:
            continue
        return True
    return False


def _span_containing(
    index: int, spans: Sequence[ProtectedSpan]
) -> Optional[ProtectedSpan]:
    for span in spans:
        if span.start <= index < span.end:
            return span
    return None
