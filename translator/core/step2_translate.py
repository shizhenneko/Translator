from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from openai.types.chat import ChatCompletionMessageParam

from .chunking import ChunkPlanEntry
from ..llm.client import KimiClient
from .validation import (
    require_bool,
    require_dict,
    require_int,
    require_str,
    require_str_list,
)
from .preservation import (
    PreservationError,
    protect,
    restore,
    validate_fence_counts,
    validate_math_delimiters,
    validate_url_targets,
)
from ..markdown.autofix import autofix_markdown
from ..markdown.normalize import normalize_markdown_for_preview


_PLACEHOLDER_TOKEN_RE = r"__([A-Z][A-Z_]*)_[0-9]{3}__"
_PLACEHOLDER_RE = re.compile(rf"(?<![_A-Za-z0-9]){_PLACEHOLDER_TOKEN_RE}")
_PLACEHOLDER_BACKTICK_RE = re.compile(rf"`+(?P<token>{_PLACEHOLDER_TOKEN_RE})`+")
_PLACEHOLDER_FENCED_BLOCK_RE = re.compile(
    "".join(
        [
            r"(?m)(^|\n)(?P<fence>`{3,}|~{3,})[A-Za-z0-9_-]*[ \t]*\n",
            rf"(?P<token>{_PLACEHOLDER_TOKEN_RE})\n",
            r"(?P=fence)[ \t]*(?=\n|$)",
        ]
    )
)
_MARKDOWN_LINK_RE = re.compile(r"!?(\[[^\]]*\]\([^)]+\))")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_PURE_LINK_LIST_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\[[^\]]+\]\([^)]+\)[ \t]*$"
)
_PAREN_ENGLISH_GLOSS_RE = re.compile(
    r"[（(][A-Za-z][A-Za-z0-9 ,，;:/_+.-]{0,80}[)）]"
)
_DOTTED_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
)
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?\b")
_ALLOWLIST_PHRASE_RE = re.compile(
    r"\b(?:Gitlet|Git|SHA|Java|JUnit|Unix|Windows|MacOS|Gradescope|Beacon|Ed|Capers|Head|HEAD|HashMap|TreeMap|Office Hours|Gitbug)\b"
)
_ALLOWLIST_WORDS = {
    "blob",
    "blobs",
    "tree",
    "trees",
    "commit",
    "commits",
    "branch",
    "branches",
    "checkout",
    "reset",
    "merge",
    "push",
    "pull",
    "fetch",
    "status",
    "log",
    "global",
    "grader",
    "graders",
    "master",
    "staging",
    "staged",
    "snapshot",
    "snapshots",
    "pattern",
    "patterns",
    "testing",
    "debugging",
    "serialization",
    "serializing",
    "deserialization",
    "deserialize",
    "serialize",
    "repository",
    "repositories",
    "remote",
    "remotes",
    "spec",
    "specification",
}
_ENGLISH_PROSE_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "actually",
    "all",
    "and",
    "before",
    "between",
    "both",
    "but",
    "can",
    "care",
    "command",
    "commands",
    "create",
    "doing",
    "does",
    "dont",
    "each",
    "explain",
    "file",
    "files",
    "final",
    "first",
    "from",
    "help",
    "here",
    "how",
    "important",
    "just",
    "know",
    "last",
    "later",
    "line",
    "lines",
    "magic",
    "matched",
    "more",
    "most",
    "need",
    "only",
    "part",
    "paste",
    "portion",
    "previous",
    "probably",
    "provided",
    "remember",
    "relevant",
    "save",
    "seems",
    "see",
    "show",
    "some",
    "specify",
    "still",
    "that",
    "the",
    "them",
    "then",
    "these",
    "they",
    "this",
    "those",
    "through",
    "time",
    "too",
    "use",
    "used",
    "using",
    "very",
    "what",
    "when",
    "which",
    "while",
    "will",
    "with",
    "work",
    "works",
    "worry",
    "would",
    "your",
}
_ENGLISH_LEAK_MAX_ATTEMPTS = 4
_LINE_REWRITE_MAX_ATTEMPTS = 2


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


_MAX_GLOSSARY_TERMS_PER_CHUNK = _read_env_int("TRANSLATOR_GLOSSARY_MAX_TERMS", 30)
_MAX_GLOSSARY_CHARS_PER_CHUNK = _read_env_int("TRANSLATOR_GLOSSARY_MAX_CHARS", 2000)
_GLOSSARY_MODES = {"filtered", "full"}
logger = logging.getLogger(__name__)


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


def _strip_placeholder_backticks(text: str) -> str:
    text = _PLACEHOLDER_BACKTICK_RE.sub(lambda m: m.group("token"), text)
    return _PLACEHOLDER_FENCED_BLOCK_RE.sub(
        lambda m: f"{m.group(1)}{m.group('token')}", text
    )


def _strip_prompt_markers(text: str) -> str:
    cleaned = re.sub(r"^[ \t]*(<<<|>>>)\s*$\n?", "", text, flags=re.MULTILINE)
    return re.sub(r"^[ \t]*(<<<|>>>)\s*(#+\s*)", r"\2", cleaned, flags=re.MULTILINE)


def _stabilize_chunk_text(text: str) -> str:
    cleaned = _strip_prompt_markers(text)
    cleaned = normalize_markdown_for_preview(cleaned)
    return autofix_markdown(cleaned)


def _normalize_glossary_text(value: str) -> str:
    normalized = value.casefold().replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize_glossary_text(value: str) -> List[str]:
    normalized = _normalize_glossary_text(value)
    if not normalized:
        return []
    return re.findall(r"[a-z0-9]+", normalized)


def _has_exact_phrase(term_normalized: str, chunk_normalized: str) -> bool:
    if not term_normalized:
        return False
    pattern = r"\b" + re.escape(term_normalized) + r"\b"
    if re.search(pattern, chunk_normalized):
        return True
    return term_normalized in chunk_normalized


def _has_word_boundary(term_normalized: str, chunk_normalized: str) -> bool:
    if not term_normalized:
        return False
    if re.search(r"[^\w\s]", term_normalized):
        return term_normalized in chunk_normalized
    pattern = r"\b" + re.escape(term_normalized) + r"\b"
    return re.search(pattern, chunk_normalized) is not None


def _filter_glossary_for_chunk(
    glossary: Sequence[Dict[str, object]],
    chunk_text: str,
    max_terms: int = _MAX_GLOSSARY_TERMS_PER_CHUNK,
    max_chars: int = _MAX_GLOSSARY_CHARS_PER_CHUNK,
) -> List[Dict[str, object]]:
    if not glossary or not chunk_text:
        return []
    if max_terms <= 0 or max_chars <= 0:
        return []

    chunk_normalized = _normalize_glossary_text(chunk_text)
    if not chunk_normalized:
        return []
    chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk_normalized))

    candidates: List[tuple[int, int, Dict[str, object], int]] = []
    for index, entry in enumerate(glossary):
        item = _require_dict(entry, f"glossary[{index}]")
        term_en = _require_str(item.get("term_en"), f"glossary[{index}].term_en")
        term_zh = _require_str(item.get("term_zh"), f"glossary[{index}].term_zh")
        note_zh = _require_str(item.get("note_zh"), f"glossary[{index}].note_zh")

        term_tokens = _tokenize_glossary_text(term_en)
        term_token_set = set(term_tokens)
        term_normalized = _normalize_glossary_text(term_en)

        priority: Optional[int] = None
        if len(term_token_set) >= 2:
            if _has_exact_phrase(term_normalized, chunk_normalized):
                priority = 1
            else:
                overlap = len(term_token_set.intersection(chunk_tokens))
                if term_token_set and overlap / len(term_token_set) >= 0.5:
                    priority = 3
        else:
            if _has_word_boundary(term_normalized, chunk_normalized):
                priority = 2

        if priority is None:
            continue

        entry_chars = len(term_en) + len(term_zh) + len(note_zh)
        candidates.append((priority, index, item, entry_chars))

    candidates.sort(key=lambda item: (item[0], item[1]))
    filtered: List[Dict[str, object]] = []
    total_chars = 0
    for _, _, entry, entry_chars in candidates:
        if len(filtered) >= max_terms:
            break
        if total_chars + entry_chars > max_chars:
            continue
        filtered.append(entry)
        total_chars += entry_chars
    return filtered


def translate_chunk(
    chunk_text: str,
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    *,
    client: Optional[KimiClient] = None,
    chunk_id: str = "",
    index: int = 0,
    style_rules: Optional[Sequence[str]] = None,
    prompt_outline_mode: str = "headings",
    glossary_mode: str = "filtered",
    output_format: str = "readable",
) -> ChunkTranslation:
    if not chunk_text:
        return ChunkTranslation(chunk_id=chunk_id, index=index, text="", warnings=[])
    if glossary_mode not in _GLOSSARY_MODES:
        raise Step2TranslateError("glossary_mode must be 'filtered' or 'full'")
    if output_format not in {"readable", "analysis"}:
        raise Step2TranslateError("output_format must be 'readable' or 'analysis'")
    started_at = time.perf_counter()

    glossary_for_chunk = glossary
    if glossary_mode == "filtered":
        glossary_for_chunk = _filter_glossary_for_chunk(glossary, chunk_text)

    protected_text, restoration_map = protect(chunk_text)
    llm_client = client or KimiClient()
    expected_placeholders = sorted(restoration_map.keys())
    translated = _translate_with_placeholder_retries(
        client=llm_client,
        outline=outline,
        glossary=glossary_for_chunk,
        protected_chunk=protected_text,
        original_chunk=chunk_text,
        restoration_map=restoration_map,
        expected_placeholders=expected_placeholders,
        style_rules=style_rules,
        prompt_outline_mode=prompt_outline_mode,
        output_format=output_format,
    )

    try:
        cleaned = _strip_placeholder_backticks(translated)
        cleaned = _strip_unknown_placeholders(cleaned, restoration_map)
        restored = restore(cleaned, restoration_map, strict=False)
    except PreservationError as exc:
        raise Step2TranslateError(f"restore failed: {exc}") from exc

    cleaned_restored = _stabilize_chunk_text(restored)
    cleaned_restored = _repair_untranslated_english_lines(
        cleaned_restored,
        client=llm_client,
    )
    remaining_leaks = _find_untranslated_english_lines(cleaned_restored)
    if remaining_leaks:
        raise Step2TranslateError(
            "translation left untranslated English prose after targeted repair: "
            + "; ".join(remaining_leaks[:3])
        )
    qa_warnings = _validate_restored_chunk(
        original=chunk_text, restored=cleaned_restored
    )
    warnings = qa_warnings + _collect_glossary_warnings(
        cleaned_restored, glossary_for_chunk
    )
    _log_chunk_progress(
        chunk_id=chunk_id,
        index=index,
        chunk_chars=len(chunk_text),
        output_format=output_format,
        warnings=len(warnings),
        elapsed_ms=int((time.perf_counter() - started_at) * 1000),
    )
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
    prompt_outline_mode: str = "headings",
    glossary_mode: str = "filtered",
    output_format: str = "readable",
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
                prompt_outline_mode=prompt_outline_mode,
                glossary_mode=glossary_mode,
                output_format=output_format,
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
    original_chunk: str,
    restoration_map: Dict[str, str],
    expected_placeholders: Sequence[str],
    style_rules: Optional[Sequence[str]],
    prompt_outline_mode: str = "headings",
    output_format: str = "readable",
) -> str:
    messages = _build_step2_messages(
        outline,
        glossary,
        protected_chunk,
        style_rules=style_rules,
        placeholder_tokens=expected_placeholders if expected_placeholders else None,
        prompt_outline_mode=prompt_outline_mode,
        output_format=output_format,
    )

    expected_set = set(expected_placeholders)
    best_result: Optional[str] = None
    best_score: Optional[Tuple[int, int]] = None
    best_leaks: List[str] = []
    max_attempts = _ENGLISH_LEAK_MAX_ATTEMPTS

    for attempt in range(max_attempts):
        if attempt > 0:
            retry_note = ""
            if best_leaks:
                retry_note = (
                    "\n\nRetry requirement:"
                    "\n- The previous draft still left untranslated English prose."
                    "\n- Translate those English sentences into Chinese while preserving inline code, commands, file names, and placeholders."
                    "\n- In particular, fix lines similar to:"
                    + "".join(f"\n  - {item}" for item in best_leaks[:3])
                )
            messages = _build_step2_messages(
                outline,
                glossary,
                protected_chunk,
                style_rules=style_rules,
                placeholder_tokens=expected_placeholders,
                prompt_outline_mode=prompt_outline_mode,
                output_format=output_format,
            )
            if retry_note:
                user_content = str(messages[-1]["content"]) + retry_note
                messages[-1] = {"role": "user", "content": user_content}
        result = client.chat_completion(messages, json_mode=False)
        missing = sum(1 for p in expected_set if p not in result)
        english_leaks = 999999
        leaks_preview: List[str] = []
        if missing == 0:
            english_leaks, leaks_preview = _count_untranslated_english_lines(
                original_chunk=original_chunk,
                translated_chunk=result,
                restoration_map=restoration_map,
            )
        if missing == 0 and english_leaks == 0:
            return result
        score = (missing, english_leaks)
        if best_score is None or score < best_score:
            best_score = score
            best_result = result
            best_leaks = leaks_preview

    if best_result is not None and best_score is not None and best_score[0] == 0:
        return best_result
    raise Step2TranslateError("translation failed after placeholder validation retries")


def _build_step2_messages(
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    protected_chunk: str,
    *,
    style_rules: Optional[Sequence[str]] = None,
    placeholder_tokens: Optional[Sequence[str]] = None,
    prompt_outline_mode: str = "full",
    output_format: str = "readable",
) -> List[ChatCompletionMessageParam]:
    readable_mode = output_format == "readable"
    system_prompt = (
        "You are a technical translation assistant for study notes. "
        "Output ONLY Markdown. Do not wrap output in JSON or code fences. "
        "Preserve all placeholders and Markdown structure exactly."
    )

    outline_block = _render_condensed_outline(outline, mode=prompt_outline_mode)
    glossary_block = _render_glossary(glossary)
    rules_block = _render_style_rules(style_rules)

    user_lines = [
        "Translate the chunk from English to Chinese.",
        "Requirements:",
        "- Output Markdown only; no JSON wrapper, no extra commentary.",
        "- Preserve Markdown structure, links, math, code fences, and inline code.",
        "- NEVER convert inline code (`backticks`) into fenced code blocks (```). Keep `code` as `code`.",
        "- Do not translate or modify placeholder tokens like __CODE_BLOCK_001__.",
        "- Term style: 首次出现使用 `中文（English）`，后续只用中文。",
        "- Glossary enforcement is soft: prefer glossary terms when relevant.",
        "- Translate explanatory English prose and Markdown terms into Chinese; keep English only for commands, filenames, identifiers, or when needed in parentheses for precision.",
    ]
    if readable_mode:
        user_lines.extend(
            [
                "- Prefer clean, natural Chinese for direct reading.",
                "- Do not add study annotations, summaries, or extra bullets unless they are required to keep the Markdown readable and faithful.",
            ]
        )
    else:
        user_lines.extend(
            [
                "- Annotation density: medium (key explanation + 1 example/analogy).",
                "- Annotation format: `> **学习批注：** ...` or `> **背景扩展：** ...`.",
            ]
        )

    if rules_block:
        user_lines.extend(["", "Style rules:", rules_block])

    if placeholder_tokens:
        user_lines.extend(["", "Placeholders (unchanged):"])
        if readable_mode:
            user_lines.append(", ".join(placeholder_tokens))
        else:
            user_lines.extend(f"- {token}" for token in placeholder_tokens)

    include_outline = (not readable_mode) or prompt_outline_mode == "full"
    if include_outline:
        user_lines.extend(["", "Condensed outline:", outline_block])

    user_lines.extend(
        [
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


def _render_condensed_outline(
    outline: Sequence[Dict[str, object]], mode: str = "full"
) -> str:
    if not outline:
        return "_No outline provided._"
    lines: List[str] = []
    for index, entry in enumerate(outline):
        item = _require_dict(entry, f"outline[{index}]")
        level = _require_int(item.get("level"), f"outline[{index}].level")
        heading = _require_str(item.get("heading"), f"outline[{index}].heading")

        if mode == "headings":
            # Headings-only mode: no summary_bullets or key_takeaways
            line = f"- L{level} {heading}"
            lines.append(line)
        else:
            # Full mode (legacy): include summary_bullets and key_takeaways
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
    for excerpt in _find_untranslated_english_lines(restored)[:5]:
        warnings.append(f"QA warning: untranslated English prose: {excerpt}")
    return warnings


def _count_untranslated_english_lines(
    *,
    original_chunk: str,
    translated_chunk: str,
    restoration_map: Dict[str, str],
) -> Tuple[int, List[str]]:
    try:
        cleaned = _strip_placeholder_backticks(translated_chunk)
        cleaned = _strip_unknown_placeholders(cleaned, restoration_map)
        restored = restore(cleaned, restoration_map, strict=False)
    except PreservationError:
        return 999999, []
    stabilized = _stabilize_chunk_text(restored)
    leaks = _find_untranslated_english_lines(stabilized)
    return len(leaks), leaks


def _repair_untranslated_english_lines(
    text: str,
    *,
    client: KimiClient,
) -> str:
    candidates = _collect_untranslated_english_line_candidates(text)
    if not candidates:
        return text

    lines = text.splitlines()
    had_trailing_newline = text.endswith("\n")
    for line_index, original_line in candidates:
        rewritten_line = _rewrite_line_in_chinese(
            original_line,
            client=client,
        )
        lines[line_index] = rewritten_line

    repaired = "\n".join(lines)
    if had_trailing_newline:
        repaired += "\n"
    return _stabilize_chunk_text(repaired)


def _rewrite_line_in_chinese(
    line: str,
    *,
    client: KimiClient,
) -> str:
    best_line = line
    best_score = len(_find_untranslated_english_lines(line))

    for attempt in range(_LINE_REWRITE_MAX_ATTEMPTS):
        messages = _build_line_rewrite_messages(line, attempt=attempt)
        rewritten = client.chat_completion(messages, json_mode=False)
        normalized = _normalize_rewritten_line(rewritten, original_line=line)
        score = len(_find_untranslated_english_lines(normalized))
        if score < best_score:
            best_line = normalized
            best_score = score
        if score == 0:
            return normalized
    return best_line


def _build_line_rewrite_messages(
    line: str,
    *,
    attempt: int,
) -> List[ChatCompletionMessageParam]:
    user_lines = [
        "Rewrite the following single Markdown line into natural Chinese.",
        "Requirements:",
        "- Return exactly one Markdown line.",
        "- Translate English explanatory prose into Chinese.",
        "- Preserve list markers, blockquote markers, inline code, links, math, and punctuation structure.",
        "- Keep commands, file names, identifiers, and placeholder-like tokens unchanged.",
        "- For Markdown terms like triple backticks or plaintext, translate them naturally into Chinese. You may keep the English term in parentheses if it improves accuracy.",
        "",
        "Line:",
        "<<<",
        line,
        ">>>",
    ]
    if attempt > 0:
        user_lines.extend(
            [
                "",
                "Retry requirement:",
                "- The previous rewrite still left too much English prose.",
                "- Translate more completely while keeping technical identifiers intact.",
            ]
        )
    return [
        {
            "role": "system",
            "content": "You are editing one Markdown line for a Chinese technical document.",
        },
        {
            "role": "user",
            "content": "\n".join(user_lines),
        },
    ]


def _normalize_rewritten_line(response: str, *, original_line: str) -> str:
    text = _strip_prompt_markers(response).strip()
    fenced = re.fullmatch(r"`{3,}[A-Za-z0-9_-]*\s*\n(.*?)\n`{3,}", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not text:
        return original_line

    original_prefix = re.match(r"^[ \t>*-]*?(?:\d+[.)][ \t]+|[-*+][ \t]+|>[ \t]+)?", original_line)
    if original_prefix and original_prefix.group(0):
        prefix = original_prefix.group(0)
        stripped_original = original_line[len(prefix) :]
        if stripped_original and not text.startswith(prefix):
            text = prefix + text.lstrip()
    return text.rstrip()


def _collect_untranslated_english_line_candidates(text: str) -> List[Tuple[int, str]]:
    findings: List[Tuple[int, str]] = []
    in_fence = False
    for line_index, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not stripped:
            continue
        if stripped.startswith("# "):
            continue
        if stripped.startswith("Source: url "):
            continue
        if _PURE_LINK_LIST_LINE_RE.fullmatch(stripped):
            continue

        scrubbed = re.sub(r"`[^`]+`", "", stripped)
        scrubbed = _MARKDOWN_LINK_RE.sub("", scrubbed)
        scrubbed = _PAREN_ENGLISH_GLOSS_RE.sub("", scrubbed)
        scrubbed = _DOTTED_IDENTIFIER_RE.sub("", scrubbed)
        person_names = _PERSON_NAME_RE.findall(scrubbed)
        scrubbed = _PERSON_NAME_RE.sub("", scrubbed)
        scrubbed = _ALLOWLIST_PHRASE_RE.sub("", scrubbed)
        words = _LATIN_WORD_RE.findall(scrubbed)
        meaningful = [word for word in words if word.casefold() not in _ALLOWLIST_WORDS]
        if len(meaningful) < 2:
            continue
        prose_words = [
            word for word in meaningful if word.casefold() in _ENGLISH_PROSE_WORDS
        ]
        if not _CHINESE_CHAR_RE.search(scrubbed):
            findings.append((line_index, line))
            continue
        if len(prose_words) >= 2:
            findings.append((line_index, line))
            continue
        if prose_words and len(" ".join(meaningful)) >= 18:
            findings.append((line_index, line))
            continue
        if len(" ".join(meaningful)) >= 40:
            if person_names and len(person_names) >= 2 and len(prose_words) <= 1:
                continue
            findings.append((line_index, line))
    return findings


def _find_untranslated_english_lines(text: str) -> List[str]:
    return [line.strip()[:160] for _, line in _collect_untranslated_english_line_candidates(text)]


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
    return require_dict(value, label, Step2TranslateError, expected="a dict")


def _require_str(value: object, label: str) -> str:
    return require_str(value, label, Step2TranslateError)


def _require_int(value: object, label: str) -> int:
    return require_int(value, label, Step2TranslateError)


def _require_bool(value: object, label: str) -> bool:
    return require_bool(value, label, Step2TranslateError)


def _require_str_list(value: object, label: str) -> List[str]:
    return require_str_list(
        value,
        label,
        Step2TranslateError,
        expected="a list of strings",
    )


def _escape_table_cell(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return escaped.replace("\n", "<br>")


def _log_chunk_progress(
    *,
    chunk_id: str,
    index: int,
    chunk_chars: int,
    output_format: str,
    warnings: int,
    elapsed_ms: int,
) -> None:
    if os.getenv("TRANSLATOR_TIMING_LOG", "1") == "0":
        return
    logger.warning(
        "chunk_done index=%s chunk_id=%s chars=%s output_format=%s warnings=%s elapsed_ms=%s",
        index,
        chunk_id or "-",
        chunk_chars,
        output_format,
        warnings,
        elapsed_ms,
    )
