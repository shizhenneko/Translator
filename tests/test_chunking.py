"""Tests for chunking layer."""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.chunking import (
    build_chunk_plan,
    reconstruct_from_chunks,
    chunk_plan_payload,
)


@pytest.fixture
def large_fixture():
    """Load the large fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "large.md"
    return fixture_path.read_text(encoding="utf-8")


def test_chunker_reconstructs_original(large_fixture):
    """Test that chunker reconstruction is byte-for-byte identical."""
    original = large_fixture

    # Build chunk plan
    chunks = build_chunk_plan(original, max_chunk_chars=2000)

    # Verify chunks were created
    assert len(chunks) > 1, "Expected multiple chunks for large fixture"

    # Reconstruct
    reconstructed = reconstruct_from_chunks(chunks)

    # Verify byte-for-byte identity
    assert reconstructed == original, "Reconstructed text must match original exactly"


def test_chunker_reconstructs_simple():
    """Test reconstruction with simple examples."""
    test_cases = [
        "Single paragraph",
        "Para 1\n\nPara 2",
        "# Heading\n\nContent",
        "# H1\n\n## H2\n\nContent",
    ]

    for original in test_cases:
        chunks = build_chunk_plan(original, max_chunk_chars=100)
        reconstructed = reconstruct_from_chunks(chunks)
        assert reconstructed == original, f"Failed for: {original}"


def test_chunk_size_limit():
    """Test that chunks respect size limit."""
    # Create text with proper blank line separators
    paragraphs = ["Content paragraph. " * 10 for _ in range(10)]
    text = "# Section\n\n" + "\n\n".join(paragraphs)
    max_size = 500

    chunks = build_chunk_plan(text, max_chunk_chars=max_size)

    # Verify all chunks are within limit
    for chunk in chunks:
        assert len(chunk.source_text) <= max_size, (
            f"Chunk exceeds limit: {len(chunk.source_text)} > {max_size}"
        )


def test_chunk_ids_sequential():
    """Test that chunk IDs are sequential."""
    # Create text with proper blank line separators
    paras1 = "\n\n".join([f"Para {i}." for i in range(20)])
    paras2 = "\n\n".join([f"Para {i}." for i in range(20, 40)])
    text = f"# Section 1\n\n{paras1}\n\n# Section 2\n\n{paras2}"
    chunks = build_chunk_plan(text, max_chunk_chars=200)

    assert len(chunks) > 1

    # Verify IDs are sequential
    for i, chunk in enumerate(chunks, start=1):
        expected_id = f"chunk-{i:04d}"
        assert chunk.chunk_id == expected_id, (
            f"Expected {expected_id}, got {chunk.chunk_id}"
        )


def test_empty_text():
    """Test chunking empty text."""
    chunks = build_chunk_plan("", max_chunk_chars=1000)
    assert len(chunks) == 0

    reconstructed = reconstruct_from_chunks(chunks)
    assert reconstructed == ""


def test_single_chunk():
    """Test text that fits in single chunk."""
    text = "Short text"
    chunks = build_chunk_plan(text, max_chunk_chars=1000)

    assert len(chunks) == 1
    assert chunks[0].source_text == text

    reconstructed = reconstruct_from_chunks(chunks)
    assert reconstructed == text


def test_chunk_at_headings():
    """Test that chunking respects heading boundaries."""
    text = "# Heading 1\n\nContent 1\n\n# Heading 2\n\nContent 2"
    chunks = build_chunk_plan(text, max_chunk_chars=50)

    # Should split at headings
    assert len(chunks) >= 2


def test_chunk_at_blank_lines():
    """Test that chunking uses blank lines as split points."""
    text = "Para 1\n\nPara 2\n\nPara 3\n\nPara 4"
    chunks = build_chunk_plan(text, max_chunk_chars=20)

    # Should split at blank lines
    assert len(chunks) > 1


def test_protected_spans_not_split():
    """Test that protected spans (code, math) are not split."""
    text = """# Section

```python
def func():
    return 1
```

More content here.
"""
    chunks = build_chunk_plan(text, max_chunk_chars=500)

    code_block = "```python\ndef func():\n    return 1\n```"

    chunks_with_code = [c for c in chunks if "```python" in c.source_text]
    assert len(chunks_with_code) == 1, "Code block should not be split"


def test_chunk_plan_payload():
    """Test chunk plan payload serialization."""
    text = "# Section\n\nContent"
    chunks = build_chunk_plan(text, max_chunk_chars=100)

    payload = chunk_plan_payload(chunks)

    assert isinstance(payload, list)
    assert len(payload) == len(chunks)

    for item in payload:
        assert "chunk_id" in item
        assert "source_text" in item
        assert "separators" in item


def test_invalid_max_chunk_chars():
    """Test that invalid max_chunk_chars raises error."""
    with pytest.raises(ValueError, match="max-chunk-chars must be positive"):
        build_chunk_plan("text", max_chunk_chars=0)

    with pytest.raises(ValueError, match="max-chunk-chars must be positive"):
        build_chunk_plan("text", max_chunk_chars=-1)


def test_paragraph_exceeds_limit():
    """Test that oversized paragraph is force-split into smaller chunks."""
    text = "x" * 1000

    chunks = build_chunk_plan(text, max_chunk_chars=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.source_text) <= 100
    assert "".join(c.source_text for c in chunks) == text


def test_separators_preserved():
    """Test that separators are preserved in chunks."""
    text = "Para 1\n\n\nPara 2\n\nPara 3"
    chunks = build_chunk_plan(text, max_chunk_chars=1000)

    # Single chunk should preserve all separators
    assert len(chunks) == 1
    assert chunks[0].source_text == text

    # Verify separators list
    assert len(chunks[0].separators) > 0


def test_reconstruction_with_multiple_separators():
    """Test reconstruction with various separator patterns."""
    text = "A\n\nB\n\n\nC\n\nD"
    chunks = build_chunk_plan(text, max_chunk_chars=1000)
    reconstructed = reconstruct_from_chunks(chunks)

    assert reconstructed == text


def test_large_fixture_multiple_chunks(large_fixture):
    """Test that large fixture creates multiple chunks."""
    chunks = build_chunk_plan(large_fixture, max_chunk_chars=1000)

    # Large fixture should create multiple chunks
    assert len(chunks) > 3, f"Expected multiple chunks, got {len(chunks)}"

    # Verify all chunks are within limit
    for chunk in chunks:
        assert len(chunk.source_text) <= 1000


def test_chunk_id_width():
    """Test that chunk ID width adjusts for large numbers."""
    # Create text that will generate many chunks
    text = "\n\n".join([f"Para {i}" for i in range(100)])
    chunks = build_chunk_plan(text, max_chunk_chars=20)

    # Should have many chunks
    assert len(chunks) > 10

    # All IDs should have consistent width
    id_lengths = [len(chunk.chunk_id) for chunk in chunks]
    assert len(set(id_lengths)) == 1, "All chunk IDs should have same length"
