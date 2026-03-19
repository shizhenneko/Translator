"""Tests for Snapdown to Mermaid conversion (TDD RED phase)."""

import sys
import importlib
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

jina_reader_fetcher = importlib.import_module("translator.jina_reader_fetcher")
snapdown_converter = importlib.import_module("translator.snapdown_converter")
SnapdownBlock = getattr(jina_reader_fetcher, "SnapdownBlock")
convert_snapdown_to_mermaid = getattr(snapdown_converter, "convert_snapdown_to_mermaid")


# Snapdown DSL fixtures (embedded as strings)
SNAPDOWN_SIMPLE_ARRAY = """cities -> (Array<string>
  0 -> (string "Boston")
  1 -> (string "New York")
  2 -> (string "San Francisco")
)"""

SNAPDOWN_NESTED_OBJECT = """user -> (Object
  name -> (string "Alice")
  age -> (number 30)
  address -> (Object
    city -> (string "Boston")
    zip -> (string "02101")
  )
)"""

SNAPDOWN_MIXED_TYPES = """data -> (Object
  items -> (Array<number>
    0 -> (number 10)
    1 -> (number 20)
    2 -> (number 30)
  )
  metadata -> (Object
    count -> (number 3)
    source -> (string "api")
  )
)"""

# Expected Mermaid outputs (for mocking)
MERMAID_SIMPLE_ARRAY = """graph TD
    cities["cities: Array&lt;string&gt;"]
    cities --> item0["0: Boston"]
    cities --> item1["1: New York"]
    cities --> item2["2: San Francisco"]"""

MERMAID_NESTED_OBJECT = """graph TD
    user["user: Object"]
    user --> name["name: Alice"]
    user --> age["age: 30"]
    user --> address["address: Object"]
    address --> city["city: Boston"]
    address --> zip["zip: 02101"]"""

MERMAID_WITH_BACKTICKS = """graph TD
    data["data: `Object`"]
    data --> value["`value: test`"]"""

MERMAID_SANITIZED = """graph TD
    data["data: Object"]
    data --> value["value: test"]"""


def test_basic_conversion():
    """Test basic Snapdown to Mermaid conversion."""
    # Arrange
    blocks = [
        SnapdownBlock(language="markdown", content="# Header"),
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
        SnapdownBlock(language="markdown", content="Some text"),
    ]

    mock_client = Mock()
    mock_client.chat_completion = MagicMock(return_value=MERMAID_SIMPLE_ARRAY)

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert
    assert len(result) == 3
    assert result[0].language == "markdown"
    assert result[0].content == "# Header"

    # The snapdown block should be converted to mermaid
    assert result[1].language == "mermaid"
    assert result[1].content == MERMAID_SIMPLE_ARRAY

    assert result[2].language == "markdown"
    assert result[2].content == "Some text"

    # Verify LLM was called with correct parameters
    mock_client.chat_completion.assert_called_once()


def test_fallback_on_conversion_failure():
    """Test that failed conversion preserves original snapdown block."""
    # Arrange
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_NESTED_OBJECT),
    ]

    mock_client = Mock()
    # Simulate conversion failure (returns empty string)
    mock_client.chat_completion = MagicMock(return_value="")

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert - original block should be preserved
    assert len(result) == 1
    assert result[0].language == "snapdown"
    assert result[0].content == SNAPDOWN_NESTED_OBJECT


def test_fallback_on_none_response():
    """Test that None response preserves original snapdown block."""
    # Arrange
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_MIXED_TYPES),
    ]

    mock_client = Mock()
    # Simulate conversion failure (returns None)
    mock_client.chat_completion = MagicMock(return_value=None)

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert - original block should be preserved
    assert len(result) == 1
    assert result[0].language == "snapdown"
    assert result[0].content == SNAPDOWN_MIXED_TYPES


def test_sanitization_removes_backticks():
    """Test that Mermaid content is sanitized to remove backticks."""
    # Arrange
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
    ]

    mock_client = Mock()
    # Return Mermaid with backticks
    mock_client.chat_completion = MagicMock(return_value=MERMAID_WITH_BACKTICKS)

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert - backticks should be removed
    assert len(result) == 1
    assert result[0].language == "mermaid"
    assert "`" not in result[0].content
    assert result[0].content == MERMAID_SANITIZED


def test_multiple_snapdown_blocks():
    """Test conversion of multiple snapdown blocks."""
    # Arrange
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
        SnapdownBlock(language="markdown", content="Separator"),
        SnapdownBlock(language="snapdown", content=SNAPDOWN_NESTED_OBJECT),
    ]

    mock_client = Mock()
    mock_client.chat_completion = MagicMock(
        side_effect=[MERMAID_SIMPLE_ARRAY, MERMAID_NESTED_OBJECT]
    )

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert
    assert len(result) == 3
    assert result[0].language == "mermaid"
    assert result[0].content == MERMAID_SIMPLE_ARRAY
    assert result[1].language == "markdown"
    assert result[2].language == "mermaid"
    assert result[2].content == MERMAID_NESTED_OBJECT

    # Verify LLM was called twice
    assert mock_client.chat_completion.call_count == 2


def test_cache_reuse_for_identical_blocks():
    """Test that identical Snapdown blocks reuse cache within a call."""
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
    ]

    mock_client = Mock()
    mock_client.chat_completion = MagicMock(return_value=MERMAID_SIMPLE_ARRAY)

    result = convert_snapdown_to_mermaid(blocks, mock_client)

    assert len(result) == 2
    assert result[0].language == "mermaid"
    assert result[1].language == "mermaid"
    assert result[0].content == MERMAID_SIMPLE_ARRAY
    assert result[1].content == MERMAID_SIMPLE_ARRAY
    assert mock_client.chat_completion.call_count == 1


def test_non_json_response_accepted():
    """Test that non-JSON responses are accepted as raw Mermaid."""
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
    ]

    raw_mermaid = "graph TD\nA-->B"
    mock_client = Mock()
    mock_client.chat_completion = MagicMock(return_value=raw_mermaid)

    result = convert_snapdown_to_mermaid(blocks, mock_client)

    assert len(result) == 1
    assert result[0].language == "mermaid"
    assert result[0].content == raw_mermaid


def test_exception_in_client_preserves_block():
    """Test that client exceptions preserve original snapdown block."""
    blocks = [
        SnapdownBlock(language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY),
    ]

    mock_client = Mock()
    mock_client.chat_completion = MagicMock(side_effect=RuntimeError("boom"))

    result = convert_snapdown_to_mermaid(blocks, mock_client)

    assert len(result) == 1
    assert result[0].language == "snapdown"
    assert result[0].content == SNAPDOWN_SIMPLE_ARRAY


def test_non_snapdown_blocks_unchanged():
    """Test that non-snapdown blocks pass through unchanged."""
    # Arrange
    blocks = [
        SnapdownBlock(language="python", content="print('hello')"),
        SnapdownBlock(language="javascript", content="console.log('world')"),
        SnapdownBlock(language="markdown", content="# Title"),
    ]

    mock_client = Mock()

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert - all blocks unchanged
    assert len(result) == 3
    assert result[0].language == "python"
    assert result[0].content == "print('hello')"
    assert result[1].language == "javascript"
    assert result[1].content == "console.log('world')"
    assert result[2].language == "markdown"
    assert result[2].content == "# Title"

    # Verify LLM was never called
    mock_client.chat_completion.assert_not_called()


def test_empty_blocks_list():
    """Test conversion with empty blocks list."""
    # Arrange
    blocks = []
    mock_client = Mock()

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert
    assert len(result) == 0
    mock_client.chat_completion.assert_not_called()


def test_preserves_heading_attribute():
    """Test that heading attribute is preserved during conversion."""
    # Arrange
    blocks = [
        SnapdownBlock(
            language="snapdown", content=SNAPDOWN_SIMPLE_ARRAY, heading="Data Structure"
        ),
    ]

    mock_client = Mock()
    mock_client.chat_completion = MagicMock(return_value=MERMAID_SIMPLE_ARRAY)

    # Act
    result = convert_snapdown_to_mermaid(blocks, mock_client)

    # Assert
    assert len(result) == 1
    assert result[0].language == "mermaid"
    assert result[0].heading == "Data Structure"
