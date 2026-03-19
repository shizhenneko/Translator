from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Set
from urllib.parse import unquote, urlparse

from ..io.fs_utils import read_text

WriteTextCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class TranslationOptions:
    jina_api_key_env: Optional[str] = None
    timeout: float = 30.0
    max_chunk_chars: int = 5000
    concurrency: int = 2
    snapdown_to_mermaid: bool = True
    prompt_outline_mode: str = "headings"
    prompt_glossary_mode: str = "filtered"
    output_format: str = "readable"


@dataclass(frozen=True)
class TranslationResult:
    url: str
    out_path: str
    success: bool
    error: Optional[str] = None

    @property
    def file_name(self) -> str:
        return os.path.basename(self.out_path)


def parse_url_list_text(content: str, source_label: str = "url list") -> List[str]:
    urls: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    if not urls:
        raise ValueError(f"no URLs found in: {source_label}")
    return urls


def read_url_list(path: str) -> List[str]:
    return parse_url_list_text(read_text(path), source_label=path)


def normalize_urls(values: Sequence[str]) -> List[str]:
    urls: List[str] = []
    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        urls.append(stripped)
    if not urls:
        raise ValueError("no URLs provided")
    return urls


def collect_url_lists(paths: Sequence[str]) -> List[str]:
    urls: List[str] = []
    for path in paths:
        urls.extend(read_url_list(path))
    if not urls:
        raise ValueError("no URLs provided")
    return urls


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or ""
    path = parsed.path or ""
    if parsed.query:
        path = f"{path}?{parsed.query}"
    raw = f"{host}{path}" if host or path else url
    raw = unquote(raw).strip().strip("/")
    if not raw:
        raw = host or "url"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    if not slug:
        slug = "url"
    return slug[:120].strip("-") or "url"


def require_out_dir(out_dir: str) -> str:
    if not out_dir:
        raise ValueError("output directory is required")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"output directory is not usable: {out_dir}")
    return out_dir


def build_single_out_path(out_dir: str, url: str) -> str:
    return os.path.join(require_out_dir(out_dir), f"{slugify_url(url)}.md")


def build_batch_out_path(
    out_dir: str, url: str, index: int, used_names: Set[str]
) -> str:
    slug = slugify_url(url)
    name = f"{index:03d}-{slug}.md"
    if name in used_names:
        counter = 2
        while name in used_names:
            name = f"{index:03d}-{slug}-{counter}.md"
            counter += 1
    used_names.add(name)
    return os.path.join(out_dir, name)


def apply_jina_api_key_env(jina_api_key_env: Optional[str]) -> None:
    if not jina_api_key_env:
        return
    api_key = os.environ.get(jina_api_key_env)
    if not api_key:
        raise ValueError(f"missing API key in env var: {jina_api_key_env}")
    os.environ["JINA_API_KEY"] = api_key


def translate_url_to_path(
    *,
    url: str,
    out_path: str,
    options: TranslationOptions,
    write_text: WriteTextCallback,
) -> str:
    apply_jina_api_key_env(options.jina_api_key_env)

    from ..core.pipeline import translate_document

    return translate_document(
        source_type="url",
        source_value=url,
        out_path=out_path,
        max_chunk_chars=options.max_chunk_chars,
        concurrency=options.concurrency,
        timeout_seconds=options.timeout,
        snapdown_to_mermaid=options.snapdown_to_mermaid,
        prompt_outline_mode=options.prompt_outline_mode,
        prompt_glossary_mode=options.prompt_glossary_mode,
        output_format=options.output_format,
        write_text=write_text,
    )


def translate_urls_batch(
    *,
    urls: Sequence[str],
    out_dir: str,
    options: TranslationOptions,
    write_text: WriteTextCallback,
    progress_callback: Optional[Callable[[TranslationResult, int, int], None]] = None,
) -> List[TranslationResult]:
    normalized_urls = normalize_urls(urls)
    target_dir = require_out_dir(out_dir)
    used_names: Set[str] = set()
    results: List[TranslationResult] = []
    total = len(normalized_urls)

    for index, url in enumerate(normalized_urls, start=1):
        out_path = build_batch_out_path(target_dir, url, index, used_names)
        try:
            _ = translate_url_to_path(
                url=url,
                out_path=out_path,
                options=options,
                write_text=write_text,
            )
            result = TranslationResult(url=url, out_path=out_path, success=True)
        except Exception as exc:
            result = TranslationResult(
                url=url,
                out_path=out_path,
                success=False,
                error=str(exc),
            )
        results.append(result)
        if progress_callback is not None:
            progress_callback(result, index, total)

    return results
