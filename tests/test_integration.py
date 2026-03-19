"""Integration tests for the end-to-end translation pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.pipeline import PipelineError, translate_document


@pytest.fixture
def sample_fixture() -> str:
    fixture_path = Path(__file__).parent / "fixtures" / "sample.md"
    return fixture_path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MOONSHOT_API_KEY")),
    reason="DEEPSEEK_API_KEY not set - skipping integration test",
)
def test_end_to_end_translation_analysis_mode(sample_fixture: str, tmp_path: Path) -> None:
    input_file = tmp_path / "input.md"
    input_file.write_text(sample_fixture, encoding="utf-8")
    output_file = tmp_path / "output.md"

    written_content: list[str] = []

    def mock_write_text(path: str, content: str) -> None:
        written_content.append(content)
        Path(path).write_text(content, encoding="utf-8")

    result = translate_document(
        source_type="file",
        source_value=str(input_file),
        out_path=str(output_file),
        max_chunk_chars=2000,
        concurrency=1,
        output_format="analysis",
        write_text=mock_write_text,
    )

    assert len(written_content) == 1
    assert output_file.exists()

    output_content = output_file.read_text(encoding="utf-8")
    assert "## Meta" in output_content
    assert "## Outline" in output_content
    assert "## Glossary" in output_content
    assert result == output_content


def test_pipeline_error_invalid_source_type() -> None:
    def mock_write_text(path: str, content: str) -> None:
        _ = (path, content)

    with pytest.raises(PipelineError, match="source_type must be"):
        translate_document(
            source_type="invalid",
            source_value="test",
            out_path="output.md",
            write_text=mock_write_text,
        )


def test_pipeline_error_empty_source_value() -> None:
    def mock_write_text(path: str, content: str) -> None:
        _ = (path, content)

    with pytest.raises(PipelineError, match="source_value is required"):
        translate_document(
            source_type="file",
            source_value="",
            out_path="output.md",
            write_text=mock_write_text,
        )
