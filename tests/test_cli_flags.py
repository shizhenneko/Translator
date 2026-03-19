import sys
import importlib
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

cli_module = importlib.import_module("translator.cli")
build_parser = getattr(cli_module, "build_parser")


def test_translate_url_default_snapdown_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "translate-url",
            "--url",
            "https://example.com",
            "--out",
            "out.md",
        ]
    )
    assert args.no_snapdown_mermaid is False
    assert args.output_format == "readable"


def test_translate_unified_default_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "translate",
            "--url",
            "https://example.com",
            "--out",
            "out.md",
        ]
    )
    assert args.no_snapdown_mermaid is False
    assert args.output_format == "readable"


def test_translate_unified_url_list_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "translate",
            "--url-list",
            "urls.txt",
            "--out-dir",
            "out",
        ]
    )
    assert args.url_list == ["urls.txt"]
    assert args.out_dir == "out"
    assert args.output_format == "readable"


def test_translate_url_disable_snapdown_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "translate-url",
            "--url",
            "https://example.com",
            "--out",
            "out.md",
            "--no-snapdown-mermaid",
        ]
    )
    assert args.no_snapdown_mermaid is True


def test_translate_url_batch_flag():
    parser = build_parser()
    args = parser.parse_args(
        [
            "translate-url-batch",
            "--url-list",
            "urls.txt",
            "--out-dir",
            "out",
            "--no-snapdown-mermaid",
        ]
    )
    assert args.no_snapdown_mermaid is True
    assert args.output_format == "readable"


def test_lint_md_renderer_flags_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "lint-md",
            "--in",
            "sample.md",
        ]
    )
    assert args.strict_renderer == "on"
    assert args.max_safe_list_depth == 1


def test_sanitize_md_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sanitize-md",
            "--in",
            "sample.md",
            "--out",
            "clean.md",
        ]
    )
    assert args.input_path == "sample.md"
    assert args.out == "clean.md"
    assert args.in_place is False
