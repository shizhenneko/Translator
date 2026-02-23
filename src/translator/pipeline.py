from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, cast

from .chunking import build_chunk_plan
from .llm_client import KimiClient
from .step1_profile import profile as profile_step1
from .step2_translate import ChunkTranslation, translate_chunks

from .jina_reader_fetcher import JinaReaderConfig, fetch_markdown


class PipelineError(RuntimeError):
    pass


def translate_document(
    *,
    source_type: str,
    source_value: str,
    out_path: str,
    max_chunk_chars: int = 8000,
    concurrency: int = 3,
    timeout_seconds: Optional[float] = None,
    title_hint: Optional[str] = None,
    client: Optional[KimiClient] = None,
    write_text: Optional[Callable[[str, str], None]] = None,
) -> str:
    if source_type not in {"url", "file"}:
        raise PipelineError("source_type must be 'url' or 'file'")
    if not source_value:
        raise PipelineError("source_value is required")

    content = _read_source(
        source_type=source_type,
        source_value=source_value,
        timeout_seconds=timeout_seconds,
    )
    content = _clean_jina_artifacts(content)

    llm_client = client or KimiClient()
    model_id = cast(str, getattr(llm_client, "_model", "unknown"))

    profile_payload, _ = profile_step1(
        content=content,
        source_type=source_type,
        source_value=source_value,
        title_hint=title_hint,
        client=llm_client,
    )

    outline = _require_list(profile_payload.get("outline"), "outline")
    glossary = _require_list(profile_payload.get("glossary"), "glossary")
    style_guide = _require_dict(profile_payload.get("style_guide"), "style_guide")
    style_rules = _require_str_list(style_guide.get("rules"), "style_guide.rules")

    chunks = build_chunk_plan(content, max_chunk_chars)
    translations = translate_chunks(
        chunks,
        cast(Sequence[Dict[str, object]], outline),
        cast(Sequence[Dict[str, object]], glossary),
        client=llm_client,
        concurrency=concurrency,
        style_rules=style_rules,
    )

    output = _assemble_output(
        source_type=source_type,
        source_value=source_value,
        model_id=model_id,
        outline=cast(Sequence[Dict[str, object]], outline),
        glossary=cast(Sequence[Dict[str, object]], glossary),
        translations=translations,
    )
    if write_text is None:
        raise PipelineError("write_text callback is required")
    write_text(out_path, output)
    return output


def _read_source(
    *,
    source_type: str,
    source_value: str,
    timeout_seconds: Optional[float],
) -> str:
    if source_type == "url":
        config = None
        if timeout_seconds is not None:
            config = JinaReaderConfig(timeout_seconds=int(timeout_seconds))
        return fetch_markdown(source_value, config=config)

    if not os.path.exists(source_value):
        raise PipelineError(f"input file not found: {source_value}")
    if not os.path.isfile(source_value):
        raise PipelineError(f"input path is not a file: {source_value}")
    with open(source_value, "r", encoding="utf-8") as handle:
        return handle.read()


_EMPTY_ANCHOR_RE = re.compile(r"\[]\(https?://[^)]+\)\s*")
_DOUBLE_FENCE_RE = re.compile(r"^(`{6,}|~{6,})$", re.MULTILINE)


def _clean_jina_artifacts(content: str) -> str:
    content = _EMPTY_ANCHOR_RE.sub("", content)
    content = _DOUBLE_FENCE_RE.sub(
        lambda m: (
            m.group(1)[: len(m.group(1)) // 2]
            + "\n"
            + m.group(1)[: len(m.group(1)) // 2]
        ),
        content,
    )
    return content


def _assemble_output(
    *,
    source_type: str,
    source_value: str,
    model_id: str,
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    translations: Sequence[ChunkTranslation],
) -> str:
    meta = _render_meta(
        source_type=source_type, source_value=source_value, model_id=model_id
    )
    outline_md = _render_outline(outline)
    glossary_md = _render_glossary(glossary)
    body = "".join(item.text for item in translations)

    sections = [meta, outline_md, glossary_md, body]
    normalized = [section.strip("\n") for section in sections]
    output = "\n\n".join(normalized).rstrip() + "\n"
    return _fix_heading_collisions(output)


def _render_meta(*, source_type: str, source_value: str, model_id: str) -> str:
    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    lines = [
        "## Meta",
        f"- Source: {source_type} {source_value}",
        f"- Timestamp: {timestamp}",
        f"- Model: {model_id}",
    ]
    return "\n".join(lines)


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
    return re.sub(
        r"^([ \t]*\d+[.)]\s+[^\n]*?)(#{1,6}\s+)",
        r"\1\n\2",
        text,
        flags=re.MULTILINE,
    )


def _render_outline(outline: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = ["## Outline"]
    if not outline:
        lines.append("_No outline entries._")
        return "\n".join(lines)

    for index, entry in enumerate(outline):
        item = _require_dict(entry, f"outline[{index}]")
        level = _require_int(item.get("level"), f"outline[{index}].level")
        heading = _require_str(item.get("heading"), f"outline[{index}].heading")
        heading_level = min(6, max(3, level + 2))
        lines.append(f"{'#' * heading_level} {heading}")

        summary_bullets = _require_str_list(
            item.get("summary_bullets"), f"outline[{index}].summary_bullets"
        )
        if summary_bullets:
            lines.append("- Summary")
            lines.extend(f"  - {bullet}" for bullet in summary_bullets)

        key_takeaways = _require_str_list(
            item.get("key_takeaways"), f"outline[{index}].key_takeaways"
        )
        if key_takeaways:
            lines.append("- Key takeaways")
            lines.extend(f"  - {bullet}" for bullet in key_takeaways)

        lines.append("")

    if lines[-1] == "":
        _ = lines.pop()
    return "\n".join(lines)


def _render_glossary(glossary: Sequence[Dict[str, object]]) -> str:
    lines: List[str] = ["## Glossary"]
    if not glossary:
        lines.append("_No glossary entries._")
        return "\n".join(lines)

    lines.append("| Term (EN) | Term (ZH) | Note (ZH) | Keep EN First Use |")
    lines.append("| --- | --- | --- | --- |")
    for index, entry in enumerate(glossary):
        item = _require_dict(entry, f"glossary[{index}]")
        term_en = _require_str(item.get("term_en"), f"glossary[{index}].term_en")
        term_zh = _require_str(item.get("term_zh"), f"glossary[{index}].term_zh")
        note_zh = _require_str(item.get("note_zh"), f"glossary[{index}].note_zh")
        keep_en = _require_bool(
            item.get("keep_en_on_first_use"),
            f"glossary[{index}].keep_en_on_first_use",
        )
        keep_value = "true" if keep_en else "false"
        lines.append(
            "| {term_en} | {term_zh} | {note_zh} | {keep_en} |".format(
                term_en=_escape_table_cell(term_en),
                term_zh=_escape_table_cell(term_zh),
                note_zh=_escape_table_cell(note_zh),
                keep_en=keep_value,
            )
        )

    return "\n".join(lines)


def _require_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise PipelineError(f"{label} must be an object")
    return cast(Dict[str, object], value)


def _require_list(value: object, label: str) -> List[object]:
    if not isinstance(value, list):
        raise PipelineError(f"{label} must be an array")
    return cast(List[object], value)


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PipelineError(f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PipelineError(f"{label} must be an integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise PipelineError(f"{label} must be a boolean")
    return value


def _require_str_list(value: object, label: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise PipelineError(f"{label} must be a list of strings")
    values: List[str] = []
    items = cast(List[object], value)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise PipelineError(f"{label}[{index}] must be a string")
        values.append(item)
    return values


def _escape_table_cell(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return escaped.replace("\n", "<br>")
