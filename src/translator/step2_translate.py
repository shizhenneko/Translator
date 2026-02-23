from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Sequence, cast

from openai.types.chat import ChatCompletionMessageParam

from .chunking import ChunkPlanEntry
from .llm_client import KimiClient
from .preservation import (
    PreservationError,
    protect,
    restore,
    validate_fence_counts,
    validate_math_delimiters,
    validate_url_targets,
)


_PLACEHOLDER_RE = re.compile(r"__([A-Z_]+)_[0-9]{3}__")


class Step2TranslateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChunkTranslation:
    chunk_id: str
    index: int
    text: str
    warnings: List[str]


def _strip_unknown_placeholders(text: str, restoration_map: Dict[str, str]) -> str:
    known = set(restoration_map.keys())
    return _PLACEHOLDER_RE.sub(
        lambda m: m.group(0) if m.group(0) in known else "", text
    )


def _strip_prompt_markers(text: str) -> str:
    cleaned = re.sub(r"^[ \t]*(<<<|>>>)\s*$\n?", "", text, flags=re.MULTILINE)
    return re.sub(r"^[ \t]*(<<<|>>>)\s*(#+\s*)", r"\2", cleaned, flags=re.MULTILINE)


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


def translate_chunk(
    chunk_text: str,
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    *,
    client: Optional[KimiClient] = None,
    chunk_id: str = "",
    index: int = 0,
    style_rules: Optional[Sequence[str]] = None,
) -> ChunkTranslation:
    if not chunk_text:
        return ChunkTranslation(chunk_id=chunk_id, index=index, text="", warnings=[])

    protected_text, restoration_map = protect(chunk_text)
    if len(restoration_map) > 30:
        protected_text, restoration_map = protect(chunk_text, skip_inline_code=True)
    llm_client = client or KimiClient()
    expected_placeholders = sorted(restoration_map.keys())
    translated = _translate_with_placeholder_retries(
        client=llm_client,
        outline=outline,
        glossary=glossary,
        protected_chunk=protected_text,
        expected_placeholders=expected_placeholders,
        style_rules=style_rules,
    )

    try:
        cleaned = _strip_unknown_placeholders(translated, restoration_map)
        restored = restore(cleaned, restoration_map, strict=False)
    except PreservationError as exc:
        raise Step2TranslateError(f"restore failed: {exc}") from exc

    cleaned_restored = _strip_prompt_markers(restored)
    cleaned_restored = _fix_heading_collisions(cleaned_restored)
    qa_warnings = _validate_restored_chunk(
        original=chunk_text, restored=cleaned_restored
    )
    warnings = qa_warnings + _collect_glossary_warnings(cleaned_restored, glossary)
    return ChunkTranslation(
        chunk_id=chunk_id, index=index, text=cleaned_restored, warnings=warnings
    )


def translate_chunks(
    chunks: Sequence[ChunkPlanEntry],
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    *,
    client: Optional[KimiClient] = None,
    concurrency: int = 3,
    style_rules: Optional[Sequence[str]] = None,
) -> List[ChunkTranslation]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if not chunks:
        return []

    results: List[Optional[ChunkTranslation]] = [None] * len(chunks)
    futures: Dict[Future[ChunkTranslation], int] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for index, chunk in enumerate(chunks):
            future = executor.submit(
                translate_chunk,
                chunk.source_text,
                outline,
                glossary,
                client=client,
                chunk_id=chunk.chunk_id,
                index=index,
                style_rules=style_rules,
            )
            futures[future] = index

        try:
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
        except Exception:
            for future in futures:
                _ = future.cancel()
            raise

    translated: List[ChunkTranslation] = []
    for item in results:
        if item is None:
            raise Step2TranslateError("missing chunk translation result")
        translated.append(item)
    return translated


def _translate_with_placeholder_retries(
    *,
    client: KimiClient,
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    protected_chunk: str,
    expected_placeholders: Sequence[str],
    style_rules: Optional[Sequence[str]],
) -> str:
    messages = _build_step2_messages(
        outline,
        glossary,
        protected_chunk,
        style_rules=style_rules,
        placeholder_tokens=expected_placeholders if expected_placeholders else None,
    )

    if not expected_placeholders:
        return client.chat_completion(messages, json_mode=False)

    expected_set = set(expected_placeholders)
    best_result: Optional[str] = None
    best_missing = len(expected_set)
    max_attempts = 3

    for attempt in range(max_attempts):
        if attempt > 0:
            messages = _build_step2_messages(
                outline,
                glossary,
                protected_chunk,
                style_rules=style_rules,
                placeholder_tokens=expected_placeholders,
            )
        result = client.chat_completion(messages, json_mode=False)
        missing = sum(1 for p in expected_set if p not in result)
        if missing == 0:
            return result
        if missing < best_missing:
            best_missing = missing
            best_result = result

    if best_result is not None:
        return best_result
    raise Step2TranslateError("translation failed after placeholder validation retries")


def _build_step2_messages(
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    protected_chunk: str,
    *,
    style_rules: Optional[Sequence[str]] = None,
    placeholder_tokens: Optional[Sequence[str]] = None,
) -> List[ChatCompletionMessageParam]:
    system_prompt = (
        "You are a technical translation assistant for study notes. "
        "Output ONLY Markdown. Do not wrap output in JSON or code fences. "
        "Preserve all placeholders and Markdown structure exactly."
    )

    outline_block = _render_condensed_outline(outline)
    glossary_block = _render_glossary(glossary)
    rules_block = _render_style_rules(style_rules)

    user_lines = [
        "Translate the chunk from English to Chinese.",
        "Requirements:",
        "- Output Markdown only; no JSON wrapper, no extra commentary.",
        "- Preserve Markdown structure, links, math, code fences, and inline code.",
        "- Do not translate or modify placeholder tokens like __CODE_BLOCK_001__.",
        "- Term style: 首次出现使用 `中文（English）`，后续只用中文。",
        "- Annotation density: medium (key explanation + 1 example/analogy).",
        "- Annotation format: `> **学习批注：** ...` or `> **背景扩展：** ...`.",
        "- Glossary enforcement is soft: prefer glossary terms when relevant.",
    ]

    if rules_block:
        user_lines.extend(["", "Style rules:", rules_block])

    if placeholder_tokens:
        user_lines.extend(
            [
                "",
                "Placeholders (must appear exactly once, unchanged):",
            ]
        )
        user_lines.extend(f"- {token}" for token in placeholder_tokens)

    user_lines.extend(
        [
            "",
            "Condensed outline:",
            outline_block,
            "",
            "Glossary:",
            glossary_block,
            "",
            "Chunk (protected text, keep placeholders unchanged):",
            "<<<",
            protected_chunk,
            ">>>",
        ]
    )

    user_prompt = "\n".join(user_lines)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _is_placeholder_validation_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "placeholder missing" in message
        or "placeholder duplicated" in message
        or "unknown placeholder" in message
    )


def _render_condensed_outline(outline: Sequence[Dict[str, object]]) -> str:
    if not outline:
        return "_No outline provided._"
    lines: List[str] = []
    for index, entry in enumerate(outline):
        item = _require_dict(entry, f"outline[{index}]")
        level = _require_int(item.get("level"), f"outline[{index}].level")
        heading = _require_str(item.get("heading"), f"outline[{index}].heading")
        summary_bullets = _require_str_list(
            item.get("summary_bullets"), f"outline[{index}].summary_bullets"
        )
        key_takeaways = _require_str_list(
            item.get("key_takeaways"), f"outline[{index}].key_takeaways"
        )

        details: List[str] = []
        if summary_bullets:
            details.append("Summary: " + "; ".join(summary_bullets))
        if key_takeaways:
            details.append("Takeaways: " + "; ".join(key_takeaways))

        line = f"- L{level} {heading}"
        if details:
            line += " | " + " | ".join(details)
        lines.append(line)
    return "\n".join(lines)


def _render_glossary(glossary: Sequence[Dict[str, object]]) -> str:
    if not glossary:
        return "_No glossary entries._"
    lines = [
        "| term_en | term_zh | note_zh | keep_en_on_first_use |",
        "| --- | --- | --- | --- |",
    ]
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


def _render_style_rules(style_rules: Optional[Sequence[str]]) -> str:
    if not style_rules:
        return ""
    rules = [rule for rule in style_rules if rule]
    if not rules:
        return ""
    return "\n".join(f"- {rule}" for rule in rules)


def _validate_restored_chunk(*, original: str, restored: str) -> List[str]:
    warnings: List[str] = []
    try:
        validate_fence_counts(original, restored)
    except PreservationError as exc:
        warnings.append(f"QA warning: {exc}")
    try:
        validate_math_delimiters(original, restored)
    except PreservationError as exc:
        warnings.append(f"QA warning: {exc}")
    try:
        validate_url_targets(original, restored)
    except PreservationError as exc:
        warnings.append(f"QA warning: {exc}")

    placeholder_match = _PLACEHOLDER_RE.search(restored)
    if placeholder_match:
        warnings.append(
            f"QA warning: leftover placeholder {placeholder_match.group(0)}"
        )
    return warnings


def _collect_glossary_warnings(
    restored: str, glossary: Sequence[Dict[str, object]]
) -> List[str]:
    warnings: List[str] = []
    for index, entry in enumerate(glossary):
        item = _require_dict(entry, f"glossary[{index}]")
        term_en = _require_str(item.get("term_en"), f"glossary[{index}].term_en")
        term_zh = _require_str(item.get("term_zh"), f"glossary[{index}].term_zh")
        if term_en in restored and term_zh not in restored:
            warnings.append(
                f"glossary term '{term_en}' missing Chinese form '{term_zh}'"
            )
    return warnings


def _require_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise Step2TranslateError(f"{label} must be a dict")
    return cast(Dict[str, object], value)


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise Step2TranslateError(f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Step2TranslateError(f"{label} must be an integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise Step2TranslateError(f"{label} must be a boolean")
    return value


def _require_str_list(value: object, label: str) -> List[str]:
    if not isinstance(value, list):
        raise Step2TranslateError(f"{label} must be a list of strings")
    items = cast(List[object], value)
    values: List[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise Step2TranslateError(f"{label}[{index}] must be a string")
        values.append(item)
    return values


def _escape_table_cell(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return escaped.replace("\n", "<br>")
