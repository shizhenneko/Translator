from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, cast

from openai.types.chat import ChatCompletionMessageParam

from .llm_client import KimiClient


_PROFILE_SCHEMA_EXAMPLE = json.dumps(
    {
        "doc": {
            "title": "...",
            "source": {"type": "url|file", "value": "..."},
            "language": {"source": "en", "target": "zh-CN"},
        },
        "outline": [
            {
                "level": 1,
                "heading": "...",
                "summary_bullets": ["..."],
                "key_takeaways": ["..."],
            }
        ],
        "glossary": [
            {
                "term_en": "...",
                "term_zh": "...",
                "note_zh": "...",
                "keep_en_on_first_use": True,
            }
        ],
        "style_guide": {
            "tone": "technical-but-friendly",
            "annotation_density": "medium",
            "rules": ["..."],
        },
    },
    ensure_ascii=True,
    indent=2,
)

_SOURCE_TYPES = {"url", "file"}


class ProfileError(RuntimeError):
    pass


def profile(
    content: str,
    source_type: str,
    source_value: str,
    title_hint: Optional[str] = None,
    source_language: str = "en",
    target_language: str = "zh-CN",
    client: Optional[KimiClient] = None,
) -> Tuple[Dict[str, object], str]:
    if not content:
        raise ProfileError("content is required")
    if source_type not in _SOURCE_TYPES:
        raise ProfileError("source_type must be 'url' or 'file'")
    if not source_value:
        raise ProfileError("source_value is required")
    if not source_language:
        raise ProfileError("source_language is required")
    if not target_language:
        raise ProfileError("target_language is required")

    messages = _build_profile_messages(
        content=content,
        source_type=source_type,
        source_value=source_value,
        title_hint=title_hint,
        source_language=source_language,
        target_language=target_language,
    )

    llm_client = client or KimiClient()
    response_text = llm_client.chat_completion(messages, json_mode=True)
    payload = _parse_profile_json(response_text)
    _apply_doc_defaults(
        payload,
        source_type=source_type,
        source_value=source_value,
        source_language=source_language,
        target_language=target_language,
        title_hint=title_hint,
    )
    _apply_glossary_defaults(payload)
    markdown = render_profile_markdown(payload)
    return payload, markdown


def render_profile_markdown(payload: Dict[str, object]) -> str:
    doc = _require_dict(payload, "doc")
    title = cast(str, doc.get("title", "")).strip() or "Profile"

    outline = cast(List[object], payload.get("outline", []))
    glossary = cast(List[object], payload.get("glossary", []))

    lines: List[str] = [f"# {title}", "", "## Outline"]

    if not outline:
        lines.append("_No outline entries._")
    else:
        for entry in outline:
            item = _require_dict(entry, "outline entry")
            level = _require_int(item.get("level"), "outline.level")
            heading = _require_str(item.get("heading"), "outline.heading")
            heading_level = min(6, max(3, level + 2))
            lines.append(f"{'#' * heading_level} {heading}")

            summary_bullets = _require_str_list(
                item.get("summary_bullets"), "outline.summary_bullets"
            )
            if summary_bullets:
                lines.append("- Summary")
                lines.extend(f"  - {bullet}" for bullet in summary_bullets)

            key_takeaways = _require_str_list(
                item.get("key_takeaways"), "outline.key_takeaways"
            )
            if key_takeaways:
                lines.append("- Key takeaways")
                lines.extend(f"  - {bullet}" for bullet in key_takeaways)

            lines.append("")

    if lines[-1] != "":
        lines.append("")

    lines.append("## Glossary")
    if not glossary:
        lines.append("_No glossary entries._")
    else:
        lines.append("| Term (EN) | Term (ZH) | Note (ZH) | Keep EN First Use |")
        lines.append("| --- | --- | --- | --- |")
        for entry in glossary:
            item = _require_dict(entry, "glossary entry")
            term_en = _require_str(item.get("term_en"), "glossary.term_en")
            term_zh = _require_str(item.get("term_zh"), "glossary.term_zh")
            note_zh = _require_str(item.get("note_zh"), "glossary.note_zh")
            keep_en = _require_bool(
                item.get("keep_en_on_first_use"), "glossary.keep_en_on_first_use"
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

    return "\n".join(lines).rstrip() + "\n"


def _build_profile_messages(
    content: str,
    source_type: str,
    source_value: str,
    title_hint: Optional[str],
    source_language: str,
    target_language: str,
) -> List[ChatCompletionMessageParam]:
    system_prompt = (
        "You are a translation profiling assistant. Output ONLY valid JSON. "
        "Do not include markdown, code fences, or extra text."
    )

    user_prompt = (
        "Create a global profile for the document."
        "\nReturn a JSON object that matches the schema exactly."
        "\nSchema example:"
        f"\n{_PROFILE_SCHEMA_EXAMPLE}"
        "\nRules:"
        "\n- Output ONLY valid JSON."
        "\n- Use double quotes for all keys and strings."
        "\n- keep_en_on_first_use must be true for every glossary entry."
        '\n- tone must be "technical-but-friendly".'
        '\n- annotation_density must be "medium".'
        "\n- style_guide.rules should be a list of short, actionable rules."
        '\n- Term style: first occurrence uses "Chinese (English)".'
        "\n- Glossary enforcement is soft: prefer glossary terms when relevant."
        "\n- If a list has no items, return an empty list."
        "\n\nSource metadata:"
        f"\n- source_type: {source_type}"
        f"\n- source_value: {source_value}"
        f"\n- source_language: {source_language}"
        f"\n- target_language: {target_language}"
    )
    if title_hint:
        user_prompt += f"\n- title_hint: {title_hint}"

    user_prompt += "\n\nDocument content:\n<<<\n"
    user_prompt += content
    user_prompt += "\n>>>"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _parse_profile_json(response_text: str) -> Dict[str, object]:
    try:
        parsed = cast(object, json.loads(response_text))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"profile response is not valid JSON: {exc}") from exc
    payload = _validate_profile_payload(parsed)
    return payload


def _validate_profile_payload(payload: object) -> Dict[str, object]:
    if not isinstance(payload, dict):
        raise ProfileError("profile JSON must be an object")

    payload_dict = cast(Dict[str, object], payload)
    doc = _require_dict(payload_dict.get("doc"), "doc")
    outline = _require_list(payload_dict.get("outline"), "outline")
    glossary = _require_list(payload_dict.get("glossary"), "glossary")
    style_guide = _require_dict(payload_dict.get("style_guide"), "style_guide")

    _validate_doc(doc)
    _validate_outline(outline)
    _validate_glossary(glossary)
    _validate_style_guide(style_guide)

    return payload_dict


def _validate_doc(doc: Dict[str, object]) -> None:
    _ = _require_str(doc.get("title"), "doc.title")
    source = _require_dict(doc.get("source"), "doc.source")
    source_type = _require_str(source.get("type"), "doc.source.type")
    if source_type not in _SOURCE_TYPES:
        raise ProfileError("doc.source.type must be 'url' or 'file'")
    _ = _require_str(source.get("value"), "doc.source.value")
    language = _require_dict(doc.get("language"), "doc.language")
    _ = _require_str(language.get("source"), "doc.language.source")
    _ = _require_str(language.get("target"), "doc.language.target")


def _validate_outline(outline: List[object]) -> None:
    for index, entry in enumerate(outline):
        item = _require_dict(entry, f"outline[{index}]")
        level = _require_int(item.get("level"), f"outline[{index}].level")
        if level <= 0:
            raise ProfileError("outline.level must be positive")
        _ = _require_str(item.get("heading"), f"outline[{index}].heading")
        item["summary_bullets"] = _require_str_list(
            item.get("summary_bullets"), f"outline[{index}].summary_bullets"
        )
        item["key_takeaways"] = _require_str_list(
            item.get("key_takeaways"), f"outline[{index}].key_takeaways"
        )


def _validate_glossary(glossary: List[object]) -> None:
    for index, entry in enumerate(glossary):
        item = _require_dict(entry, f"glossary[{index}]")
        _ = _require_str(item.get("term_en"), f"glossary[{index}].term_en")
        _ = _require_str(item.get("term_zh"), f"glossary[{index}].term_zh")
        _ = _require_str(item.get("note_zh"), f"glossary[{index}].note_zh")
        _ = _require_bool(
            item.get("keep_en_on_first_use"),
            f"glossary[{index}].keep_en_on_first_use",
        )


def _validate_style_guide(style_guide: Dict[str, object]) -> None:
    tone = _require_str(style_guide.get("tone"), "style_guide.tone")
    if tone != "technical-but-friendly":
        raise ProfileError("style_guide.tone must be 'technical-but-friendly'")
    density = _require_str(
        style_guide.get("annotation_density"), "style_guide.annotation_density"
    )
    if density != "medium":
        raise ProfileError("style_guide.annotation_density must be 'medium'")
    _ = _require_str_list(style_guide.get("rules"), "style_guide.rules")


def _apply_doc_defaults(
    payload: Dict[str, object],
    source_type: str,
    source_value: str,
    source_language: str,
    target_language: str,
    title_hint: Optional[str],
) -> None:
    doc = _require_dict(payload.get("doc"), "doc")
    source = _require_dict(doc.get("source"), "doc.source")
    language = _require_dict(doc.get("language"), "doc.language")

    source["type"] = source_type
    source["value"] = source_value
    language["source"] = source_language
    language["target"] = target_language

    title = _require_str(doc.get("title"), "doc.title").strip()
    if not title and title_hint:
        doc["title"] = title_hint


def _apply_glossary_defaults(payload: Dict[str, object]) -> None:
    glossary = _require_list(payload.get("glossary"), "glossary")
    for index, entry in enumerate(glossary):
        item = _require_dict(entry, f"glossary[{index}]")
        keep_en = _require_bool(
            item.get("keep_en_on_first_use"),
            f"glossary[{index}].keep_en_on_first_use",
        )
        if not keep_en:
            item["keep_en_on_first_use"] = True


def _require_dict(value: object, label: str) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ProfileError(f"{label} must be a JSON object")
    return cast(Dict[str, object], value)


def _require_list(value: object, label: str) -> List[object]:
    if not isinstance(value, list):
        raise ProfileError(f"{label} must be a JSON array")
    return cast(List[object], value)


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ProfileError(f"{label} must be a string")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProfileError(f"{label} must be an integer")
    return value


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileError(f"{label} must be a boolean")
    return value


def _require_str_list(value: object, label: str) -> List[str]:
    # LLM may return None, a single string, or a list — coerce gracefully.
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        raise ProfileError(f"{label} must be a list of strings")
    values: List[str] = []
    items = cast(List[object], value)
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ProfileError(f"{label}[{index}] must be a string")
        values.append(item)
    return values


def _escape_table_cell(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return escaped.replace("\n", "<br>")
