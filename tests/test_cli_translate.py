from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import translator.cli as cli


def test_translate_url_creates_parent_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    written: list[tuple[str, str]] = []

    def fake_translate_document(**kwargs: object) -> str:
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, "# Output\n")
        written.append((str(out_path), "# Output\n"))
        return "# Output\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    output_path = tmp_path / "nested" / "out.md"
    exit_code = cli.main(
        [
            "translate-url",
            "--url",
            "https://example.com",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "# Output\n"
    assert written


def test_translate_unified_command_with_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    written: list[tuple[str, str]] = []

    def fake_translate_document(**kwargs: object) -> str:
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, "# Unified URL\n")
        written.append((str(out_path), "# Unified URL\n"))
        return "# Unified URL\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    output_path = tmp_path / "unified" / "out.md"
    exit_code = cli.main(
        [
            "translate",
            "--url",
            "https://example.com",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "# Unified URL\n"
    assert written


def test_translate_unified_command_with_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    written: list[tuple[str, str]] = []

    def fake_translate_document(**kwargs: object) -> str:
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, "# Unified File\n")
        written.append((str(out_path), "# Unified File\n"))
        return "# Unified File\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    input_path = tmp_path / "sample.md"
    input_path.write_text("# Sample\n", encoding="utf-8")
    output_path = tmp_path / "unified-file" / "out.md"
    exit_code = cli.main(
        [
            "translate",
            "--in",
            str(input_path),
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "# Unified File\n"
    assert written


def test_translate_unified_command_with_url_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[str] = []

    def fake_translate_document(**kwargs: object) -> str:
        calls.append(str(kwargs["source_value"]))
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, f"# {kwargs['source_value']}\n")
        return f"# {kwargs['source_value']}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/a\nhttps://example.com/b\n", encoding="utf-8")
    out_dir = tmp_path / "batch-unified"

    exit_code = cli.main(
        [
            "translate",
            "--url-list",
            str(url_file),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert out_dir.is_dir()
    assert len(list(out_dir.glob("*.md"))) == 2
    assert calls == ["https://example.com/a", "https://example.com/b"]


def test_translate_unified_command_with_repeated_urls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[str] = []

    def fake_translate_document(**kwargs: object) -> str:
        calls.append(str(kwargs["source_value"]))
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, f"# {kwargs['source_value']}\n")
        return f"# {kwargs['source_value']}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    out_dir = tmp_path / "batch-inline"
    exit_code = cli.main(
        [
            "translate",
            "--url",
            "https://example.com/a",
            "--url",
            "https://example.com/b",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert out_dir.is_dir()
    assert len(list(out_dir.glob("*.md"))) == 2
    assert calls == ["https://example.com/a", "https://example.com/b"]


def test_translate_url_batch_creates_output_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[str] = []

    def fake_translate_document(**kwargs: object) -> str:
        calls.append(str(kwargs["source_value"]))
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, f"# {kwargs['source_value']}\n")
        return f"# {kwargs['source_value']}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com/a\nhttps://example.com/b\n", encoding="utf-8")
    out_dir = tmp_path / "batch"

    exit_code = cli.main(
        [
            "translate-url-batch",
            "--url-list",
            str(url_file),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert out_dir.is_dir()
    assert len(list(out_dir.glob("*.md"))) == 2
    assert calls == ["https://example.com/a", "https://example.com/b"]


def test_translate_url_batch_returns_success_if_any_url_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    def fake_translate_document(**kwargs: object) -> str:
        source_value = str(kwargs["source_value"])
        if source_value.endswith("/fail"):
            raise RuntimeError("boom")
        write_text = kwargs["write_text"]
        out_path = kwargs["out_path"]
        write_text(out_path, "# ok\n")
        return "# ok\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)

    url_file = tmp_path / "urls.txt"
    url_file.write_text(
        "https://example.com/success\nhttps://example.com/fail\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch"

    exit_code = cli.main(
        [
            "translate-url-batch",
            "--url-list",
            str(url_file),
            "--out-dir",
            str(out_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "https://example.com/fail" in captured.err
    assert len(list(out_dir.glob("*.md"))) == 1
