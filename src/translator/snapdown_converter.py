from __future__ import annotations

import json
from hashlib import sha1
from typing import Dict, List, Optional, cast

from openai.types.chat import ChatCompletionMessageParam

from .jina_reader_fetcher import SnapdownBlock
from .llm_client import KimiClient


def _build_messages(content: str) -> List[ChatCompletionMessageParam]:
    system_prompt = (
        "You convert Snapdown DSL diagrams into Mermaid graph syntax. "
        "Return a JSON object with a single key 'mermaid' and a string value. "
        "The Mermaid value must be raw Mermaid code only with no backticks, "
        "no code fences, and no surrounding commentary."
    )
    user_prompt = f"Snapdown DSL:\n{content}"
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def _extract_mermaid(response: object) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        response_map = cast(Dict[str, object], response)
        mermaid = response_map.get("mermaid")
        return mermaid if isinstance(mermaid, str) else None
    if not isinstance(response, str):
        return None
    text = response.strip()
    if not text:
        return None
    try:
        payload = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        payload_map = cast(Dict[str, object], payload)
        mermaid = payload_map.get("mermaid")
        return mermaid if isinstance(mermaid, str) else None
    return None


def _strip_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    if lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines)


def _sanitize_mermaid(content: str) -> str:
    cleaned = _strip_fences(content)
    cleaned = cleaned.replace("`", "")
    return cleaned.strip()


def convert_snapdown_to_mermaid(
    blocks: List[SnapdownBlock],
    client: KimiClient,
) -> List[SnapdownBlock]:
    converted: List[SnapdownBlock] = []
    if not blocks:
        return converted

    cache: Dict[str, str] = {}

    for block in blocks:
        if block.language != "snapdown":
            converted.append(block)
            continue

        content_hash = sha1(block.content.encode("utf-8")).hexdigest()
        cached_mermaid = cache.get(content_hash)
        if cached_mermaid is None:
            mermaid_content: Optional[str] = None
            try:
                response = client.chat_completion(
                    _build_messages(block.content),
                    json_mode=True,
                )
                mermaid_content = _extract_mermaid(response)
            except Exception:
                mermaid_content = None

            if mermaid_content is not None:
                mermaid_content = _sanitize_mermaid(mermaid_content)
            if mermaid_content:
                cache[content_hash] = mermaid_content
                cached_mermaid = mermaid_content

        if cached_mermaid:
            converted.append(
                SnapdownBlock(
                    language="mermaid",
                    content=cached_mermaid,
                    heading=block.heading,
                )
            )
        else:
            converted.append(block)

    return converted
