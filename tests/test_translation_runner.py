from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.translation_runner import (
    TranslationOptions,
    parse_url_list_text,
    translate_urls_batch,
)


def test_parse_url_list_text_ignores_comments_and_blank_lines() -> None:
    urls = parse_url_list_text(
        "\n# comment\nhttps://example.com/a\n\n https://example.com/b \n",
        source_label="urls.txt",
    )

    assert urls == ["https://example.com/a", "https://example.com/b"]


def test_translate_urls_batch_reports_partial_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_translate_document(**kwargs: object) -> str:
        source_value = str(kwargs["source_value"])
        out_path = str(kwargs["out_path"])
        write_text = kwargs["write_text"]
        if source_value.endswith("/fail"):
            raise RuntimeError("boom")
        write_text(out_path, f"# {source_value}\n")
        return f"# {source_value}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    results = translate_urls_batch(
        urls=[
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/fail",
        ],
        out_dir=str(tmp_path / "batch"),
        options=TranslationOptions(concurrency=1),
        write_text=lambda path, content: Path(path).write_text(content, encoding="utf-8"),
    )

    assert [result.success for result in results] == [True, True, False]
    assert results[2].error == "boom"
    assert Path(results[0].out_path).exists()
    assert Path(results[1].out_path).exists()
    assert not Path(results[2].out_path).exists()
    assert len({Path(result.out_path).name for result in results}) == 3
