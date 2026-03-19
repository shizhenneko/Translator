# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.jina_reader_fetcher import _fix_jina_list_codeblocks


def test_fix_jina_list_codeblocks_indents_fence():
    markdown = "* **Usage**:\n\n```\ncmd\n```"
    expected = "* **Usage**:\n\n    ```\n    cmd\n    ```"
    assert _fix_jina_list_codeblocks(markdown) == expected


def test_fix_jina_list_codeblocks_keeps_existing_indent():
    markdown = "* Item\n\n    ```\n    cmd\n    ```"
    assert _fix_jina_list_codeblocks(markdown) == markdown
