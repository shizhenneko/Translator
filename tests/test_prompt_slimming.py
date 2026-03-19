# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import sys
from pathlib import Path
from typing import Dict, List, Optional, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.step2_translate import (
    Step2TranslateError,
    _build_step2_messages,
    _filter_glossary_for_chunk,
    _find_untranslated_english_lines,
    _repair_untranslated_english_lines,
    _render_condensed_outline,
    translate_chunk,
)


class DummyClient:
    def __init__(self) -> None:
        self.last_messages: Optional[List[Dict[str, str]]] = None

    def chat_completion(self, messages: List[Dict[str, str]], json_mode=False) -> str:
        self.last_messages = messages
        _ = json_mode
        protected = _extract_protected_chunk(messages)
        return protected.replace("The front end is here.", "前端在这里。")


class SequenceClient:
    def __init__(self, responses: List[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat_completion(self, messages: List[Dict[str, str]], json_mode=False) -> str:
        _ = messages
        _ = json_mode
        if self.calls >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self.calls]
        self.calls += 1
        return response


class MappingClient:
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping

    def chat_completion(self, messages: List[Dict[str, str]], json_mode=False) -> str:
        _ = json_mode
        user_content = messages[-1]["content"]
        if "Chunk (protected text, keep placeholders unchanged):" in user_content:
            return _extract_protected_chunk(messages)
        if "Line:" not in user_content:
            raise AssertionError("line rewrite markers missing in prompt")
        line = user_content.split("<<<", 1)[1].split(">>>", 1)[0].strip("\n")
        return self.mapping.get(line, line)


def _extract_protected_chunk(messages: List[Dict[str, str]]) -> str:
    user_content = messages[-1]["content"]
    if "<<<" not in user_content or ">>>" not in user_content:
        raise AssertionError("protected chunk markers missing in prompt")
    return user_content.split("<<<", 1)[1].split(">>>", 1)[0].strip("\n")


@pytest.fixture
def sample_outline():
    return [
        {
            "level": 1,
            "heading": "Intro",
            "summary_bullets": ["Scope"],
            "key_takeaways": ["Know goal"],
        },
        {
            "level": 2,
            "heading": "Details",
            "summary_bullets": ["Flow", "Inputs"],
            "key_takeaways": ["Check edges"],
        },
    ]


@pytest.fixture
def sample_glossary():
    return [
        {
            "term_en": "front-end",
            "term_zh": "zh-front",
            "note_zh": "UI side",
            "keep_en_on_first_use": False,
        },
        {
            "term_en": "backend",
            "term_zh": "zh-back",
            "note_zh": "API side",
            "keep_en_on_first_use": True,
        },
    ]


def test_render_outline_headings_only(sample_outline):
    rendered = _render_condensed_outline(sample_outline, mode="headings")
    assert "Summary:" not in rendered
    assert "Takeaways:" not in rendered
    assert "- L1 Intro" in rendered


def test_render_outline_full(sample_outline):
    rendered = _render_condensed_outline(sample_outline, mode="full")
    assert "Summary:" in rendered
    assert "Takeaways:" in rendered


def test_render_outline_empty():
    rendered = _render_condensed_outline([], mode="headings")
    assert rendered == "_No outline provided._"


def test_glossary_filtering_match_rules():
    glossary = [
        {"term_en": "Term", "term_zh": "zh-term", "note_zh": "A"},
        {"term_en": "front-end", "term_zh": "zh-front", "note_zh": "B"},
        {"term_en": "test", "term_zh": "zh-test", "note_zh": "C"},
        {
            "term_en": "neural network",
            "term_zh": "zh-neural",
            "note_zh": "D",
        },
        {"term_en": "noise", "term_zh": "zh-noise", "note_zh": "E"},
    ]
    chunk = "A TERM appears in front end systems. testing neural models and network designs."
    filtered = _filter_glossary_for_chunk(glossary, chunk)
    terms = [entry["term_en"] for entry in filtered]

    assert "Term" in terms
    assert "front-end" in terms
    assert "neural network" in terms
    assert "test" not in terms
    assert "noise" not in terms


def test_glossary_filtering_caps_max_terms():
    glossary = [
        {
            "term_en": "neural network",
            "term_zh": "zh-neural",
            "note_zh": "A",
        },
        {"term_en": "alpha", "term_zh": "zh-alpha", "note_zh": "B"},
        {"term_en": "beta", "term_zh": "zh-beta", "note_zh": "C"},
        {"term_en": "gamma", "term_zh": "zh-gamma", "note_zh": "D"},
    ]
    chunk = "neural network alpha beta gamma"
    filtered = _filter_glossary_for_chunk(glossary, chunk, max_terms=2)

    assert [entry["term_en"] for entry in filtered] == [
        "neural network",
        "alpha",
    ]


def test_glossary_filtering_caps_max_chars_skips_oversize():
    glossary = [
        {"term_en": "alpha", "term_zh": "zh-alpha", "note_zh": "short"},
        {
            "term_en": "beta",
            "term_zh": "zh-beta",
            "note_zh": "x" * 200,
        },
        {"term_en": "gamma", "term_zh": "zh-gamma", "note_zh": "tiny"},
    ]
    chunk = "alpha beta gamma"
    filtered = _filter_glossary_for_chunk(glossary, chunk, max_terms=5, max_chars=60)

    assert [entry["term_en"] for entry in filtered] == ["alpha", "gamma"]


def test_glossary_filtering_empty_chunk():
    glossary = [{"term_en": "alpha", "term_zh": "zh-alpha", "note_zh": "A"}]
    assert _filter_glossary_for_chunk(glossary, "") == []


def test_prompt_size_reduction_material():
    outline: List[Dict[str, object]] = []
    for i in range(12):
        summary_bullets = [
            f"Point {i}-a with extra context for sizing",
            f"Point {i}-b with extra context for sizing",
            f"Point {i}-c with extra context for sizing",
            f"Point {i}-d with extra context for sizing",
            f"Point {i}-e with extra context for sizing",
        ]
        key_takeaways = [
            f"Takeaway {i}-1 with more explanation",
            f"Takeaway {i}-2 with more explanation",
            f"Takeaway {i}-3 with more explanation",
        ]
        outline.append(
            {
                "level": 2,
                "heading": f"Section {i}",
                "summary_bullets": summary_bullets,
                "key_takeaways": key_takeaways,
            }
        )
    glossary: List[Dict[str, object]] = []
    for i in range(60):
        glossary.append(
            {
                "term_en": f"term{i}",
                "term_zh": f"z{i}",
                "note_zh": "note",
                "keep_en_on_first_use": False,
            }
        )

    full_messages = cast(
        List[Dict[str, str]],
        _build_step2_messages(outline, glossary, "chunk", prompt_outline_mode="full"),
    )
    headings_messages = cast(
        List[Dict[str, str]],
        _build_step2_messages(
            outline, glossary, "chunk", prompt_outline_mode="headings"
        ),
    )
    full_len = len(full_messages[-1]["content"])
    headings_len = len(headings_messages[-1]["content"])

    assert "Summary:" in full_messages[-1]["content"]
    assert "Summary:" not in headings_messages[-1]["content"]
    assert headings_len <= full_len * 0.7


def test_toggle_defaults_produce_slim_prompt(sample_outline, sample_glossary):
    client = DummyClient()
    translate_chunk(
        "The front end is here.",
        sample_outline,
        sample_glossary,
        client=client,
    )

    assert client.last_messages is not None
    user_prompt = client.last_messages[-1]["content"]
    assert "Summary:" not in user_prompt
    assert "Takeaways:" not in user_prompt
    assert "front-end" in user_prompt
    assert "backend" not in user_prompt


def test_find_untranslated_english_lines_flags_mixed_prose():
    text = (
        "If you’re doing a checkout command, you need to use the SHA identifier "
        "来指定检出的提交。But remember we used patterns, so we don’t actually know "
        "the SHA identifier at the time of creating the test.\n"
    )
    findings = _find_untranslated_english_lines(text)
    assert findings
    assert "But remember we used patterns" in findings[0]


def test_translate_chunk_retries_when_untranslated_english_prose_remains(sample_outline):
    english_leak = (
        "The final thing you can do with these patterns is save a matched portion. "
        "If you’re doing a `checkout` command, you need to use the SHA identifier "
        "来指定检出的提交。\n"
    )
    translated = (
        "你还能用这些模式做的最后一件事，是“保存”一段匹配到的内容。"
        "如果你正在执行 `checkout` 命令，就需要使用 SHA 标识符来指定要检出到或检出自哪个提交。\n"
    )
    client = SequenceClient([english_leak, translated])
    result = translate_chunk(
        "The final thing you can do with these patterns is save a matched portion.\n",
        sample_outline,
        [],
        client=client,
    )
    assert "The final thing you can do" not in result.text
    assert "你还能用这些模式做的最后一件事" in result.text
    assert client.calls == 2


def test_translate_chunk_raises_when_english_prose_remains_after_retries(sample_outline):
    english_leak = (
        "The final thing you can do with these patterns is save a matched portion. "
        "But remember we used patterns, so we don’t actually know the SHA identifier.\n"
    )
    client = SequenceClient([english_leak, english_leak, english_leak, english_leak])
    with pytest.raises(Step2TranslateError, match="untranslated English prose"):
        translate_chunk(
            "The final thing you can do with these patterns is save a matched portion.\n",
            sample_outline,
            [],
            client=client,
        )


def test_repair_untranslated_english_lines_rewrites_mixed_line():
    original = (
        "本规范的大部分内容将描述 triple backticks (plaintext)，which we will use to show commands.\n"
    )
    repaired = _repair_untranslated_english_lines(
        original,
        client=MappingClient(
            {
                "本规范的大部分内容将描述 triple backticks (plaintext)，which we will use to show commands.": (
                    "本规范的大部分内容将介绍三重反引号（triple backticks，plaintext），我们会用它来展示命令。"
                )
            }
        ),
    )

    assert "which we will use to show commands" not in repaired
    assert "三重反引号" in repaired
    assert not _find_untranslated_english_lines(repaired)


def test_find_untranslated_english_lines_ignores_acknowledgement_name_list():
    text = (
        "感谢 Alicia Luengo、Josh Hug、Sarah Kim、Austin Chen、Andrew Huang、Yan Zhao、"
        "Matthew Chow，尤其感谢 Alan Yao、Daniel Nguyen 和 Armani Ferrante 提供帮助。\n"
    )
    assert _find_untranslated_english_lines(text) == []


def test_toggle_legacy_modes_restore_full_prompt(sample_outline, sample_glossary):
    client = DummyClient()
    translate_chunk(
        "The front end is here.",
        sample_outline,
        sample_glossary,
        client=client,
        prompt_outline_mode="full",
        glossary_mode="full",
    )

    assert client.last_messages is not None
    user_prompt = client.last_messages[-1]["content"]
    assert "Summary:" in user_prompt
    assert "Takeaways:" in user_prompt
    assert "front-end" in user_prompt
    assert "backend" in user_prompt
