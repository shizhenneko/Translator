from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, cast

from .chunking import ChunkPlanEntry, build_chunk_plan
from .llm_client import KimiClient
from .snapdown_converter import convert_snapdown_to_mermaid
from .step1_profile import build_lightweight_profile, profile as profile_step1
from .validation import (
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
    require_str_list,
)
from .step2_translate import ChunkTranslation, translate_chunks

from .jina_reader_fetcher import (
    JinaReaderConfig,
    fetch_markdown,
    fetch_snapdown_blocks,
    insert_snapdown_blocks,
)
from .markdown_autofix import MarkdownAutofixOptions, autofix_markdown
from .markdown_lint import MarkdownLintOptions, format_issue_report, lint_markdown
from .markdown_normalize import normalize_markdown_for_preview
from .markdown_sanitize import sanitize_markdown_input

logger = logging.getLogger(__name__)


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
    snapdown_to_mermaid: bool = True,
    prompt_outline_mode: str = "headings",
    prompt_glossary_mode: str = "filtered",
    output_format: str = "readable",
    client: Optional[KimiClient] = None,
    write_text: Optional[Callable[[str, str], None]] = None,
) -> str:
    if source_type not in {"url", "file"}:
        raise PipelineError("source_type must be 'url' or 'file'")
    if not source_value:
        raise PipelineError("source_value is required")
    if output_format not in {"readable", "analysis"}:
        raise PipelineError("output_format must be 'readable' or 'analysis'")

    effective_outline_mode = prompt_outline_mode
    effective_glossary_mode = prompt_glossary_mode
    if output_format == "readable":
        effective_outline_mode = "headings"
        effective_glossary_mode = "filtered"

    llm_client = client or KimiClient()
    content = _read_source(
        source_type=source_type,
        source_value=source_value,
        timeout_seconds=timeout_seconds,
        client=llm_client,
        snapdown_to_mermaid=snapdown_to_mermaid,
    )
    content = sanitize_markdown_input(content, aggressive=True)
    content = _clean_jina_artifacts(content)
    content = normalize_markdown_for_preview(content, source_type=source_type)
    content = autofix_markdown(content, options=_guardrail_options()[0])
    model_id = cast(str, getattr(llm_client, "_model", "unknown"))

    profile_started = time.perf_counter()
    profile_payload, _ = _build_profile_payload(
        content=content,
        source_type=source_type,
        source_value=source_value,
        title_hint=title_hint,
        output_format=output_format,
        client=llm_client,
    )
    _log_stage_duration(
        "profile",
        profile_started,
        source_type=source_type,
        source_value=source_value,
        output_format=output_format,
    )

    outline = _require_list(profile_payload.get("outline"), "outline")
    glossary = _require_list(profile_payload.get("glossary"), "glossary")
    style_guide = _require_dict(profile_payload.get("style_guide"), "style_guide")
    style_rules = _require_str_list(style_guide.get("rules"), "style_guide.rules")
    doc_title = _extract_doc_title(
        profile_payload, source_value=source_value, title_hint=title_hint
    )

    chunk_started = time.perf_counter()
    chunks = _build_chunks(
        content=content,
        max_chunk_chars=max_chunk_chars,
        merge_small_sections=(output_format == "readable"),
    )
    _log_stage_duration(
        "chunk_plan",
        chunk_started,
        source_type=source_type,
        source_value=source_value,
        chunks=len(chunks),
        max_chunk_chars=max_chunk_chars,
        merge_small_sections=(output_format == "readable"),
    )
    translate_started = time.perf_counter()
    translations = _translate_chunks(
        chunks=chunks,
        outline=cast(Sequence[Dict[str, object]], outline),
        glossary=cast(Sequence[Dict[str, object]], glossary),
        client=llm_client,
        concurrency=concurrency,
        style_rules=style_rules,
        prompt_outline_mode=effective_outline_mode,
        glossary_mode=effective_glossary_mode,
        output_format=output_format,
    )
    _log_stage_duration(
        "translate_chunks",
        translate_started,
        source_type=source_type,
        source_value=source_value,
        chunks=len(chunks),
        concurrency=concurrency,
        output_format=output_format,
    )

    output = _assemble_output(
        source_type=source_type,
        source_value=source_value,
        title=doc_title,
        model_id=model_id,
        outline=cast(Sequence[Dict[str, object]], outline),
        glossary=cast(Sequence[Dict[str, object]], glossary),
        translations=translations,
        chunks=chunks,
        output_format=output_format,
    )
    output = enforce_markdown_guardrails(
        output,
        title=doc_title,
        source_type=source_type,
    )
    if write_text is None:
        raise PipelineError("write_text callback is required")
    write_text(out_path, output)
    return output


def _build_profile_payload(
    *,
    content: str,
    source_type: str,
    source_value: str,
    title_hint: Optional[str],
    output_format: str,
    client: KimiClient,
) -> tuple[Dict[str, object], str]:
    if output_format == "readable":
        return build_lightweight_profile(
            content=content,
            source_type=source_type,
            source_value=source_value,
            title_hint=title_hint,
        )
    return profile_step1(
        content=content,
        source_type=source_type,
        source_value=source_value,
        title_hint=title_hint,
        client=client,
    )


def _build_chunks(
    *,
    content: str,
    max_chunk_chars: int,
    merge_small_sections: bool,
) -> List[ChunkPlanEntry]:
    try:
        return build_chunk_plan(
            content,
            max_chunk_chars,
            merge_small_sections=merge_small_sections,
        )
    except TypeError as exc:
        if "merge_small_sections" not in str(exc):
            raise
        return build_chunk_plan(content, max_chunk_chars)


def _translate_chunks(
    *,
    chunks: Sequence[ChunkPlanEntry],
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    client: KimiClient,
    concurrency: int,
    style_rules: Sequence[str],
    prompt_outline_mode: str,
    glossary_mode: str,
    output_format: str,
) -> List[ChunkTranslation]:
    try:
        return translate_chunks(
            chunks,
            outline,
            glossary,
            client=client,
            concurrency=concurrency,
            style_rules=style_rules,
            prompt_outline_mode=prompt_outline_mode,
            glossary_mode=glossary_mode,
            output_format=output_format,
        )
    except TypeError as exc:
        if "output_format" not in str(exc):
            raise
        return translate_chunks(
            chunks,
            outline,
            glossary,
            client=client,
            concurrency=concurrency,
            style_rules=style_rules,
            prompt_outline_mode=prompt_outline_mode,
            glossary_mode=glossary_mode,
        )


def _log_stage_duration(
    stage: str,
    started_at: float,
    **metadata: object,
) -> None:
    if os.getenv("TRANSLATOR_TIMING_LOG", "1") == "0":
        return
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    meta = " ".join(f"{key}={value}" for key, value in metadata.items())
    logger.warning("pipeline_timing stage=%s elapsed_ms=%s %s", stage, elapsed_ms, meta)


def _read_source(
    *,
    source_type: str,
    source_value: str,
    timeout_seconds: Optional[float],
    client: Optional[KimiClient] = None,
    snapdown_to_mermaid: bool = True,
) -> str:
    if source_type == "url":
        config = None
        if timeout_seconds is not None:
            config = JinaReaderConfig(timeout_seconds=int(timeout_seconds))
        content = fetch_markdown(source_value, config=config)
        snapdown_blocks = fetch_snapdown_blocks(source_value, config=config)
        snapdown_blocks = (
            convert_snapdown_to_mermaid(snapdown_blocks, client)
            if client and snapdown_to_mermaid
            else snapdown_blocks
        )
        return insert_snapdown_blocks(content, snapdown_blocks)

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
    title: str,
    model_id: str,
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    translations: Sequence[ChunkTranslation],
    chunks: Sequence[ChunkPlanEntry],
    output_format: str = "readable",
) -> str:
    title_md = _render_title(title)
    body = _merge_translations(translations, chunks)
    source_line = _render_source_line(source_type=source_type, source_value=source_value)

    if output_format == "analysis":
        meta = _render_meta(
            source_type=source_type, source_value=source_value, model_id=model_id
        )
        outline_md = _render_outline(outline)
        glossary_md = _render_glossary(glossary)
        sections = [title_md, meta, outline_md, glossary_md, body]
    else:
        sections = [title_md, source_line, body]
    normalized = [section.strip("\n") for section in sections]
    output = "\n\n".join(normalized).rstrip() + "\n"
    return normalize_markdown_for_preview(
        output,
        title=title,
        source_type=source_type,
    )


def enforce_markdown_guardrails(
    markdown: str,
    *,
    title: Optional[str] = None,
    source_type: Optional[str] = None,
) -> str:
    autofix_options, lint_options = _guardrail_options()
    fixed = sanitize_markdown_input(markdown, aggressive=True)
    fixed = normalize_markdown_for_preview(
        fixed,
        title=title,
        source_type=source_type,
    )
    fixed = autofix_markdown(fixed, options=autofix_options)
    issues = lint_markdown(fixed, options=lint_options)
    if issues:
        # Retry once after aggressive stabilization to keep pipeline fail-safe.
        retried = normalize_markdown_for_preview(
            fixed,
            title=title,
            source_type=source_type,
        )
        retried = autofix_markdown(retried, options=autofix_options)
        retried_issues = lint_markdown(retried, options=lint_options)
        if len(retried_issues) <= len(issues):
            fixed = retried
            issues = retried_issues
    if issues:
        report = format_issue_report(issues)
        raise PipelineError(f"markdown guardrails failed:\n{report}")
    return fixed


def _guardrail_options() -> tuple[MarkdownAutofixOptions, MarkdownLintOptions]:
    strict_renderer = _read_env_bool("TRANSLATOR_STRICT_RENDERER", True)
    max_safe_list_depth = _read_env_int(
        "TRANSLATOR_MAX_SAFE_LIST_DEPTH", default=1, minimum=1
    )
    return (
        MarkdownAutofixOptions(
            strict_renderer=strict_renderer, max_safe_list_depth=max_safe_list_depth
        ),
        MarkdownLintOptions(
            strict_renderer=strict_renderer, max_safe_list_depth=max_safe_list_depth
        ),
    )


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


def _merge_translations(
    translations: Sequence[ChunkTranslation],
    chunks: Sequence[ChunkPlanEntry],
) -> str:
    if len(translations) != len(chunks):
        raise PipelineError("translation count does not match chunk count")
    parts: List[str] = []
    for translation, chunk in zip(translations, chunks):
        text = translation.text
        trailing = _trailing_separator(chunk)
        if trailing:
            text = text.rstrip("\n") + trailing
        parts.append(text)
    return "".join(parts)


def _trailing_separator(chunk: ChunkPlanEntry) -> str:
    if not chunk.separators:
        return ""
    return chunk.separators[-1]


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


def _render_source_line(*, source_type: str, source_value: str) -> str:
    return f"Source: {source_type} {source_value}"


def _render_title(title: str) -> str:
    compact = re.sub(r"\s+", " ", title).strip()
    compact = compact.lstrip("#").strip()
    if not compact:
        compact = "Document"
    return f"# {compact}"


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
    return require_dict(value, label, PipelineError, expected="an object")


def _require_list(value: object, label: str) -> List[object]:
    return require_list(value, label, PipelineError, expected="an array")


def _require_str(value: object, label: str) -> str:
    return require_str(value, label, PipelineError)


def _require_int(value: object, label: str) -> int:
    return require_int(value, label, PipelineError)


def _require_bool(value: object, label: str) -> bool:
    return require_bool(value, label, PipelineError)


def _require_str_list(value: object, label: str) -> List[str]:
    return require_str_list(
        value,
        label,
        PipelineError,
        allow_none=True,
        allow_str=True,
        expected="a list of strings",
    )


def _escape_table_cell(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return escaped.replace("\n", "<br>")


def _extract_doc_title(
    profile_payload: Dict[str, object],
    *,
    source_value: str,
    title_hint: Optional[str],
) -> str:
    default_title = title_hint or os.path.basename(source_value) or "Document"
    doc = profile_payload.get("doc")
    if not isinstance(doc, dict):
        return default_title
    title = doc.get("title")
    if not isinstance(title, str):
        return default_title
    stripped = title.strip()
    return stripped if stripped else default_title
