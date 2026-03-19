from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import translator.pipeline as pipeline
from translator.chunking import ChunkPlanEntry
from translator.step2_translate import ChunkTranslation


class _DummyClient:
    _model = "dummy-model"


def test_assemble_output_starts_with_h1_title():
    output = pipeline._assemble_output(
        source_type="file",
        source_value="input.md",
        title="Gitlet Notes",
        model_id="dummy-model",
        outline=[],
        glossary=[],
        translations=[
            ChunkTranslation(chunk_id="chunk-0001", index=0, text="Translated body", warnings=[])
        ],
        chunks=[
            ChunkPlanEntry(chunk_id="chunk-0001", source_text="body", separators=[])
        ],
        output_format="readable",
    )
    first_non_blank = next(line for line in output.splitlines() if line.strip())
    assert first_non_blank == "# Gitlet Notes"
    assert "## Meta" not in output
    assert "## Outline" not in output
    assert "## Glossary" not in output
    assert "Source: file input.md" in output


def test_assemble_output_analysis_mode_keeps_metadata_sections():
    output = pipeline._assemble_output(
        source_type="file",
        source_value="input.md",
        title="Gitlet Notes",
        model_id="dummy-model",
        outline=[],
        glossary=[],
        translations=[
            ChunkTranslation(chunk_id="chunk-0001", index=0, text="Translated body", warnings=[])
        ],
        chunks=[
            ChunkPlanEntry(chunk_id="chunk-0001", source_text="body", separators=[])
        ],
        output_format="analysis",
    )
    assert "## Meta" in output
    assert "## Outline" in output
    assert "## Glossary" in output


def test_enforce_markdown_guardrails_repairs_safe_patterns():
    broken = (
        "# Example\n\n"
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n"
    )
    fixed = pipeline.enforce_markdown_guardrails(broken)
    assert "```plaintext" in fixed
    assert fixed.startswith("# Example\n")


def test_enforce_markdown_guardrails_preserves_url_specific_title_cleanup():
    broken = (
        "# Project 2: Gitlet | CS 61B Spring 2021\n\n"
        "Source: url https://example.com\n\n"
        "项目 2: Gitlet\n"
        "## 关于本说明的说明\n"
    )
    fixed = pipeline.enforce_markdown_guardrails(
        broken,
        title="Project 2: Gitlet | CS 61B Spring 2021",
        source_type="url",
    )
    assert "\n项目 2: Gitlet\n" not in fixed
    assert "## 关于本说明的说明" in fixed


def test_enforce_markdown_guardrails_flattens_deep_list_fences():
    broken = (
        "# Example\n\n"
        "* Parent\n"
        "    * Child\n"
        "        ```plaintext\n"
        "        cmd\n"
        "        ```\n"
    )
    fixed = pipeline.enforce_markdown_guardrails(broken)
    assert "\n    ```plaintext\n" in fixed
    assert "        ```plaintext" not in fixed


def test_enforce_markdown_guardrails_raises_for_unbalanced_fence():
    broken = "# Broken\n\n```python\nprint('x')\n"
    with pytest.raises(pipeline.PipelineError, match="FENCE_UNBALANCED"):
        _ = pipeline.enforce_markdown_guardrails(broken)


def test_translate_document_does_not_write_when_lint_errors_remain(monkeypatch, tmp_path):
    monkeypatch.setattr(
        pipeline,
        "_read_source",
        lambda **kwargs: "source markdown",
    )
    monkeypatch.setattr(
        pipeline,
        "profile_step1",
        lambda **kwargs: (
            {
                "doc": {"title": "Title"},
                "outline": [],
                "glossary": [],
                "style_guide": {"rules": []},
            },
            "",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_chunk_plan",
        lambda content, max_chunk_chars: [
            ChunkPlanEntry(chunk_id="chunk-0001", source_text="source", separators=[])
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "translate_chunks",
        lambda *args, **kwargs: [
            ChunkTranslation(
                chunk_id="chunk-0001",
                index=0,
                text="Broken fence:\n```python\nprint('x')\n",
                warnings=[],
            )
        ],
    )

    writes: list[tuple[str, str]] = []

    def _record_write(path: str, content: str) -> None:
        writes.append((path, content))

    with pytest.raises(pipeline.PipelineError, match="FENCE_UNBALANCED"):
        _ = pipeline.translate_document(
            source_type="file",
            source_value="ignored.md",
            out_path=str(tmp_path / "out.md"),
            write_text=_record_write,
            client=_DummyClient(),
        )
    assert writes == []


def test_translate_document_sanitizes_source_before_chunking(monkeypatch, tmp_path):
    source = (
        "[](https://example.com/spec)Why specifications?\n"
        "Yes (missing answer)\n"
        "check explain\n"
        "requires: x effects: y```ts\n"
        "code\n"
        "```\n"
    )
    monkeypatch.setattr(pipeline, "_read_source", lambda **kwargs: source)
    monkeypatch.setattr(
        pipeline,
        "profile_step1",
        lambda **kwargs: (
            {
                "doc": {"title": "Title"},
                "outline": [],
                "glossary": [],
                "style_guide": {"rules": []},
            },
            "",
        ),
    )

    captured_content = {"value": ""}

    def _capture_chunk_plan(content: str, max_chunk_chars: int) -> list[ChunkPlanEntry]:
        captured_content["value"] = content
        return [ChunkPlanEntry(chunk_id="chunk-0001", source_text=content, separators=[])]

    monkeypatch.setattr(pipeline, "build_chunk_plan", _capture_chunk_plan)
    monkeypatch.setattr(
        pipeline,
        "translate_chunks",
        lambda *args, **kwargs: [
            ChunkTranslation(
                chunk_id="chunk-0001",
                index=0,
                text=captured_content["value"],
                warnings=[],
            )
        ],
    )
    writes: list[tuple[str, str]] = []
    _ = pipeline.translate_document(
        source_type="file",
        source_value="ignored.md",
        out_path=str(tmp_path / "out.md"),
        write_text=lambda path, content: writes.append((path, content)),
        client=_DummyClient(),
    )
    sanitized = captured_content["value"]
    assert "[Why specifications?](https://example.com/spec)" in sanitized
    assert "check explain" not in sanitized
    assert "missing answer" not in sanitized.lower()
    assert "\neffects: y\n```ts\n" in sanitized
    assert writes


def test_translate_document_autofixes_source_before_chunking(monkeypatch, tmp_path):
    source = (
        "* Item\n"
        "    Needs this block:\n"
        "```plaintext\n"
        "message\n"
        "```\n"
    )
    monkeypatch.setattr(pipeline, "_read_source", lambda **kwargs: source)
    monkeypatch.setattr(
        pipeline,
        "profile_step1",
        lambda **kwargs: (
            {
                "doc": {"title": "Title"},
                "outline": [],
                "glossary": [],
                "style_guide": {"rules": []},
            },
            "",
        ),
    )

    captured_content = {"value": ""}

    def _capture_chunk_plan(content: str, max_chunk_chars: int) -> list[ChunkPlanEntry]:
        captured_content["value"] = content
        return [ChunkPlanEntry(chunk_id="chunk-0001", source_text=content, separators=[])]

    monkeypatch.setattr(pipeline, "build_chunk_plan", _capture_chunk_plan)
    monkeypatch.setattr(
        pipeline,
        "translate_chunks",
        lambda *args, **kwargs: [
            ChunkTranslation(
                chunk_id="chunk-0001",
                index=0,
                text=captured_content["value"],
                warnings=[],
            )
        ],
    )

    _ = pipeline.translate_document(
        source_type="file",
        source_value="ignored.md",
        out_path=str(tmp_path / "out.md"),
        write_text=lambda path, content: None,
        client=_DummyClient(),
    )

    assert "    ```plaintext" in captured_content["value"]
    assert "\n```plaintext\n" not in captured_content["value"]


def test_translate_document_readable_mode_omits_analysis_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_read_source", lambda **kwargs: "# Title\n\nBody\n")
    profile_calls = {"count": 0}

    def _unexpected_profile(**kwargs):
        profile_calls["count"] += 1
        raise AssertionError("readable mode should use lightweight profile")

    monkeypatch.setattr(pipeline, "profile_step1", _unexpected_profile)
    monkeypatch.setattr(
        pipeline,
        "build_chunk_plan",
        lambda content, max_chunk_chars: [
            ChunkPlanEntry(chunk_id="chunk-0001", source_text="Body\n", separators=[])
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "translate_chunks",
        lambda *args, **kwargs: [
            ChunkTranslation(
                chunk_id="chunk-0001",
                index=0,
                text="Body\n",
                warnings=[],
            )
        ],
    )

    written: list[str] = []
    result = pipeline.translate_document(
        source_type="file",
        source_value="ignored.md",
        out_path=str(tmp_path / "out.md"),
        output_format="readable",
        write_text=lambda path, content: written.append(content),
        client=_DummyClient(),
    )

    assert "## Meta" not in result
    assert "## Outline" not in result
    assert "## Glossary" not in result
    assert "Source: file ignored.md" in result
    assert written == [result]
    assert profile_calls["count"] == 0


def test_translate_document_analysis_mode_uses_step1_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "_read_source", lambda **kwargs: "# Title\n\nBody\n")
    profile_calls = {"count": 0}

    def _fake_profile(**kwargs):
        profile_calls["count"] += 1
        return (
            {
                "doc": {"title": "Title"},
                "outline": [],
                "glossary": [],
                "style_guide": {"rules": []},
            },
            "",
        )

    monkeypatch.setattr(pipeline, "profile_step1", _fake_profile)
    monkeypatch.setattr(
        pipeline,
        "build_chunk_plan",
        lambda content, max_chunk_chars: [
            ChunkPlanEntry(chunk_id="chunk-0001", source_text="Body\n", separators=[])
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "translate_chunks",
        lambda *args, **kwargs: [
            ChunkTranslation(
                chunk_id="chunk-0001",
                index=0,
                text="Body\n",
                warnings=[],
            )
        ],
    )

    _ = pipeline.translate_document(
        source_type="file",
        source_value="ignored.md",
        out_path=str(tmp_path / "out.md"),
        output_format="analysis",
        write_text=lambda path, content: None,
        client=_DummyClient(),
    )

    assert profile_calls["count"] == 1
