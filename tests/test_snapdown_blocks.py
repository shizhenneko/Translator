import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.jina_reader_fetcher import (
    SnapdownBlock,
    extract_snapdown_blocks_from_html,
    append_snapdown_blocks,
    insert_snapdown_blocks,
)


def test_extract_snapdown_blocks_from_html():
    html = """
    <div>
      <script type="application/snapdown">
      cities -> (string &quot;Boston&quot;)
      </script>
      <script type="application/snapdown+json">{"x":1}</script>
    </div>
    """
    blocks = extract_snapdown_blocks_from_html(html)
    assert [block.language for block in blocks] == ["snapdown"]
    assert blocks[0].content == 'cities -> (string "Boston")'


def test_append_snapdown_blocks_adds_section():
    markdown = "# Title\n\nBody"
    blocks = [SnapdownBlock(language="snapdown", content="a -> (b)")]
    result = append_snapdown_blocks(markdown, blocks)
    assert "## Snapdown Diagrams (extracted)" in result
    assert "```snapdown" in result
    assert "a -> (b)" in result


def test_append_snapdown_blocks_uses_safe_fence():
    markdown = "# Title"
    blocks = [SnapdownBlock(language="snapdown", content="line ``` inside")]
    result = append_snapdown_blocks(markdown, blocks)
    assert "````snapdown" in result


def test_insert_snapdown_blocks_near_heading():
    html = """
    <h2>Arrays, Maps, and Sets</h2>
    <script type="application/snapdown">arr -> (Array)</script>
    <h2>Maps</h2>
    <script type="application/snapdown">map -> (Map)</script>
    """
    blocks = extract_snapdown_blocks_from_html(html)
    markdown = "## Arrays, Maps, and Sets\n\nIntro\n\n## Maps\n\nBody"
    result = insert_snapdown_blocks(markdown, blocks)
    first_index = result.index("## Arrays, Maps, and Sets")
    assert "arr -> (Array)" in result[first_index:]
    second_index = result.index("## Maps")
    assert "map -> (Map)" in result[second_index:]
