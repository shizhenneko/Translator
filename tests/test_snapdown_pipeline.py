import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, cast

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass(frozen=True)
class FakeSnapdownBlock:
    language: str
    content: str
    heading: Optional[str] = None


@dataclass(frozen=True)
class FakeChunkTranslation:
    chunk_id: str
    index: int
    text: str
    warnings: List[str]


class PipelineModule(Protocol):
    fetch_markdown: Callable[[str, Optional[object]], str]
    fetch_snapdown_blocks: Callable[
        [str, Optional[object]], Sequence["FakeSnapdownBlock"]
    ]
    profile_step1: Callable[..., Tuple[Dict[str, object], str]]
    translate_chunks: Callable[..., Sequence["FakeChunkTranslation"]]

    def translate_document(
        self,
        *,
        source_type: str,
        source_value: str,
        out_path: str,
        max_chunk_chars: int = 8000,
        concurrency: int = 3,
        timeout_seconds: Optional[float] = None,
        title_hint: Optional[str] = None,
        snapdown_to_mermaid: bool = True,
        output_format: str = "readable",
        client: Optional[object] = None,
        write_text: Optional[Callable[[str, str], None]] = None,
    ) -> str: ...


pipeline = cast(
    PipelineModule, cast(object, importlib.import_module("translator.pipeline"))
)


class FakeKimiClient:
    def chat_completion(
        self,
        messages: Sequence[Dict[str, object]],
        json_mode: bool = False,
        **kwargs: object,
    ) -> str:
        _ = (messages, json_mode, kwargs)
        return '{"mermaid": "graph TD\nA-->B"}'


def _fake_profile_step1(**kwargs: object) -> Tuple[Dict[str, object], str]:
    _ = kwargs
    payload: Dict[str, object] = {
        "outline": [
            {
                "heading": "Intro",
                "level": 1,
                "summary_bullets": [],
                "key_takeaways": [],
            }
        ],
        "glossary": [],
        "style_guide": {"rules": []},
    }
    return payload, ""


def _fake_translate_chunks(
    chunks: Sequence[object],
    outline: Sequence[Dict[str, object]],
    glossary: Sequence[Dict[str, object]],
    *,
    client: Optional[object] = None,
    concurrency: int = 1,
    style_rules: Optional[Sequence[str]] = None,
    prompt_outline_mode: str = "headings",
    glossary_mode: str = "filtered",
) -> List[FakeChunkTranslation]:
    _ = (
        outline,
        glossary,
        client,
        concurrency,
        style_rules,
        prompt_outline_mode,
        glossary_mode,
    )
    translations: List[FakeChunkTranslation] = []
    for index, chunk in enumerate(chunks):
        chunk_id = getattr(chunk, "chunk_id", str(index))
        source_text = getattr(chunk, "source_text", "")
        translations.append(
            FakeChunkTranslation(
                chunk_id=chunk_id,
                index=index,
                text=source_text,
                warnings=[],
            )
        )
    return translations


def _make_fake_fetchers():
    markdown = "# Title\n\nBody"

    def fake_fetch_markdown(url: str, config: Optional[object] = None) -> str:
        _ = (url, config)
        return markdown

    def fake_fetch_snapdown_blocks(
        url: str, config: Optional[object] = None
    ) -> List[FakeSnapdownBlock]:
        _ = (url, config)
        return [
            FakeSnapdownBlock(language="snapdown", content="A -> B", heading="Title")
        ]

    return fake_fetch_markdown, fake_fetch_snapdown_blocks


@pytest.mark.parametrize(
    "snapdown_to_mermaid, expected_fence",
    [(True, "```mermaid"), (False, "```snapdown")],
)
def test_snapdown_conversion_toggle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapdown_to_mermaid: bool,
    expected_fence: str,
):
    fake_fetch_markdown, fake_fetch_snapdown_blocks = _make_fake_fetchers()
    monkeypatch.setattr(pipeline, "fetch_markdown", fake_fetch_markdown)
    monkeypatch.setattr(pipeline, "fetch_snapdown_blocks", fake_fetch_snapdown_blocks)
    monkeypatch.setattr(pipeline, "profile_step1", _fake_profile_step1)
    monkeypatch.setattr(pipeline, "translate_chunks", _fake_translate_chunks)

    written: List[str] = []

    def fake_write_text(path: str, content: str) -> None:
        written.append(content)
        _ = Path(path).write_text(content, encoding="utf-8")

    output_path: Path = tmp_path / "out.md"
    result = pipeline.translate_document(
        source_type="url",
        source_value="https://example.com",
        out_path=str(output_path),
        max_chunk_chars=2000,
        concurrency=1,
        timeout_seconds=5,
        snapdown_to_mermaid=snapdown_to_mermaid,
        client=FakeKimiClient(),
        write_text=fake_write_text,
    )

    assert written
    assert expected_fence in result
