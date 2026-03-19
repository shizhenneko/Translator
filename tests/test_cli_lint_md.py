# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.cli import main


def test_lint_md_reports_issue_with_rule_id_and_line(tmp_path, capsys):
    input_path = tmp_path / "broken.md"
    input_path.write_text(
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n",
        encoding="utf-8",
    )

    exit_code = main(["lint-md", "--in", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "FENCE_INLINE" in captured.err
    assert "line 1" in captured.err


def test_lint_md_fix_out_writes_clean_file(tmp_path):
    input_path = tmp_path / "broken.md"
    fixed_path = tmp_path / "fixed.md"
    input_path.write_text(
        "* **Failure cases**: prints an error ```plaintext\n"
        "Found no commit with that message.\n"
        "```\n",
        encoding="utf-8",
    )

    fix_exit = main(
        [
            "lint-md",
            "--in",
            str(input_path),
            "--fix",
            "--out",
            str(fixed_path),
        ]
    )
    lint_exit = main(["lint-md", "--in", str(fixed_path)])

    assert fix_exit == 0
    assert fixed_path.exists()
    assert lint_exit == 0


def test_lint_md_fix_requires_explicit_output_mode(tmp_path, capsys):
    input_path = tmp_path / "broken.md"
    input_path.write_text("# Title\n", encoding="utf-8")

    exit_code = main(["lint-md", "--in", str(input_path), "--fix"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "--fix requires --out or --in-place" in captured.err


def test_lint_md_strict_renderer_off_allows_deep_list_fence(tmp_path):
    input_path = tmp_path / "deep-list.md"
    input_path.write_text(
        "* Parent\n"
        "    * Child\n"
        "        ```plaintext\n"
        "        cmd\n"
        "        ```\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "lint-md",
            "--in",
            str(input_path),
            "--strict-renderer",
            "off",
        ]
    )
    assert exit_code == 0


def test_lint_md_rejects_non_positive_max_safe_list_depth(tmp_path, capsys):
    input_path = tmp_path / "sample.md"
    input_path.write_text("# Title\n", encoding="utf-8")
    exit_code = main(
        [
            "lint-md",
            "--in",
            str(input_path),
            "--max-safe-list-depth",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--max-safe-list-depth must be positive" in captured.err


def test_lint_md_fix_runs_sanitize_before_autofix(tmp_path):
    input_path = tmp_path / "broken-link.md"
    fixed_path = tmp_path / "fixed-link.md"
    input_path.write_text(
        "[](https://example.com/spec)Why specifications?\n",
        encoding="utf-8",
    )
    lint_exit = main(["lint-md", "--in", str(input_path)])
    fix_exit = main(
        [
            "lint-md",
            "--in",
            str(input_path),
            "--fix",
            "--out",
            str(fixed_path),
        ]
    )
    relint_exit = main(["lint-md", "--in", str(fixed_path)])
    fixed = fixed_path.read_text(encoding="utf-8")
    assert lint_exit == 1
    assert fix_exit == 0
    assert relint_exit == 0
    assert fixed.startswith("[Why specifications?](https://example.com/spec)")


def test_sanitize_md_writes_cleaned_file(tmp_path):
    input_path = tmp_path / "raw.md"
    output_path = tmp_path / "clean.md"
    input_path.write_text(
        "[](https://example.com/spec)Why specifications?\n"
        "Yes (missing answer)\n"
        "check explain\n",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "sanitize-md",
            "--in",
            str(input_path),
            "--out",
            str(output_path),
        ]
    )
    cleaned = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "check explain" not in cleaned.lower()
    assert "missing answer" not in cleaned.lower()
    assert "[Why specifications?](https://example.com/spec)" in cleaned
