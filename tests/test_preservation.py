"""Tests for preservation layer (protect/restore)."""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.preservation import (
    protect,
    restore,
    validate_restoration,
    validate_fence_counts,
    validate_math_delimiters,
    validate_url_targets,
    PreservationError,
)


@pytest.fixture
def protection_fixture():
    """Load the protection fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "protection.md"
    return fixture_path.read_text(encoding="utf-8")


def test_protect_restore_roundtrip(protection_fixture):
    """Test that protect+restore is byte-for-byte identical."""
    original = protection_fixture

    # Protect the text
    protected, restoration_map = protect(original)

    # Verify placeholders were created
    assert len(restoration_map) > 0, "Expected placeholders to be created"

    # Restore the text
    restored = restore(protected, restoration_map)

    # Verify byte-for-byte identity
    assert restored == original, "Restored text must match original exactly"


def test_protect_restore_roundtrip_simple():
    """Test roundtrip with simple examples."""
    test_cases = [
        "Simple text with `inline code`",
        "Math: $E=mc^2$ and $$\\int x dx$$",
        "Link: [text](https://example.com)",
        "```python\ncode\n```",
        "HTML: <span>text</span>",
    ]

    for original in test_cases:
        protected, restoration_map = protect(original)
        restored = restore(protected, restoration_map)
        assert restored == original, f"Failed for: {original}"


def test_placeholder_mismatch_fails():
    """Test that QA detects placeholder mismatch."""
    original = "Text with `code` and $math$"
    protected, restoration_map = protect(original)

    # Remove a placeholder from the protected text
    placeholder = list(restoration_map.keys())[0]
    corrupted = protected.replace(placeholder, "CORRUPTED")

    # Validation should fail
    with pytest.raises(PreservationError, match="placeholder missing"):
        validate_restoration(corrupted, restoration_map)


def test_placeholder_duplication_fails():
    """Test that QA detects placeholder duplication."""
    original = "Text with `code`"
    protected, restoration_map = protect(original)

    # Duplicate a placeholder
    placeholder = list(restoration_map.keys())[0]
    corrupted = protected + " " + placeholder

    # Validation should fail
    with pytest.raises(PreservationError, match="placeholder duplicated"):
        validate_restoration(corrupted, restoration_map)


def test_unknown_placeholder_fails():
    """Test that QA detects unknown placeholders."""
    original = "Text with `code`"
    protected, restoration_map = protect(original)

    # Add an unknown placeholder
    corrupted = protected + " __UNKNOWN_001__"

    # Validation should fail
    with pytest.raises(PreservationError, match="unknown placeholder"):
        validate_restoration(corrupted, restoration_map)


def test_fence_count_validation():
    """Test that QA detects fence count mismatch."""
    original = "```python\ncode\n```"
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)

    # Validation should pass for correct restoration
    validate_fence_counts(original, restored)

    # Validation should fail if fence is removed
    corrupted = restored.replace("```", "")
    with pytest.raises(PreservationError, match="code fence count mismatch"):
        validate_fence_counts(original, corrupted)


def test_math_delimiter_validation():
    """Test that QA detects math delimiter mismatch."""
    original = "Inline: $x$ and display: $$y$$"
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)

    # Validation should pass for correct restoration
    validate_math_delimiters(original, restored)

    # Validation should fail if delimiter is removed
    corrupted = restored.replace("$", "", 1)
    with pytest.raises(PreservationError, match="math delimiter count mismatch"):
        validate_math_delimiters(original, corrupted)


def test_url_target_validation():
    """Test that QA detects URL target changes."""
    original = "[link](https://example.com)"
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)

    # Validation should pass for correct restoration
    validate_url_targets(original, restored)

    # Validation should fail if URL is changed
    corrupted = restored.replace("example.com", "different.com")
    with pytest.raises(PreservationError, match="URL target mismatch"):
        validate_url_targets(original, corrupted)


def test_empty_text():
    """Test protection of empty text."""
    original = ""
    protected, restoration_map = protect(original)

    assert protected == ""
    assert len(restoration_map) == 0

    restored = restore(protected, restoration_map)
    assert restored == original


def test_text_without_protected_elements():
    """Test text with no elements to protect."""
    original = "Just plain text without any special elements."
    protected, restoration_map = protect(original)

    assert protected == original
    assert len(restoration_map) == 0

    restored = restore(protected, restoration_map)
    assert restored == original


def test_nested_protection():
    """Test nested protected elements."""
    # Code containing math-like syntax
    original = "```python\nx = $100  # dollar sign\n```"
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)

    assert restored == original


def test_escaped_elements():
    """Test that escaped elements are not protected."""
    original = r"Escaped: \$100 and \`not code\`"
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)

    assert restored == original


def test_multiple_same_type():
    """Test multiple elements of the same type."""
    original = "`code1` and `code2` and `code3`"
    protected, restoration_map = protect(original)

    # Should have 3 placeholders
    assert len(restoration_map) == 3

    restored = restore(protected, restoration_map)
    assert restored == original


def test_preexisting_placeholder_fails():
    """Test that text with placeholder-like tokens fails."""
    original = "Text with __CODE_BLOCK_001__ placeholder"

    with pytest.raises(PreservationError, match="placeholder-like token"):
        protect(original)


def test_invalid_restoration_map():
    """Test that invalid restoration map is rejected."""
    protected = "Some text"
    invalid_map = {"invalid_key": "value"}

    with pytest.raises(PreservationError, match="invalid placeholder format"):
        restore(protected, invalid_map)


def test_underscore_adjacent_to_inline_code():
    original = "Use _`variable`_ for emphasis."
    protected, restoration_map = protect(original)
    restored = restore(protected, restoration_map)
    assert restored == original


def test_unclosed_backtick_before_code_fence():
    original = "Some `text here\n\n```python\nx = 1\n```\n\nAnd then `more` stuff.\n"
    protected, restoration_map = protect(original)
    assert "__CODE_BLOCK_001__" in protected
    restored = restore(protected, restoration_map)
    assert restored == original
