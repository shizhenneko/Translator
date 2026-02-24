import argparse
import json
import os
import re
import sys
import tempfile
from typing import Callable, Dict, List, Optional, Sequence, Set, cast
from urllib.parse import unquote, urlparse

import requests
from dotenv import load_dotenv
from .chunking import (
    ChunkPlanEntry,
    build_chunk_plan,
    chunk_plan_payload,
    reconstruct_from_chunks,
)
from .preservation import PreservationError, protect, restore
from .step1_profile import profile as profile_step1


def read_text(path: str) -> str:
    if not path:
        raise ValueError("input path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(f"input file not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"input path is not a file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _read_url_list(path: str) -> List[str]:
    content = read_text(path)
    urls: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    if not urls:
        raise ValueError(f"no URLs found in: {path}")
    return urls


def _collect_url_lists(paths: Sequence[str]) -> List[str]:
    urls: List[str] = []
    for path in paths:
        urls.extend(_read_url_list(path))
    if not urls:
        raise ValueError("no URLs provided")
    return urls


def _slugify_url(url: str) -> str:
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


def _build_batch_out_path(
    out_dir: str, url: str, index: int, used_names: Set[str]
) -> str:
    slug = _slugify_url(url)
    name = f"{index:03d}-{slug}.md"
    if name in used_names:
        counter = 2
        while name in used_names:
            name = f"{index:03d}-{slug}-{counter}.md"
            counter += 1
    used_names.add(name)
    return os.path.join(out_dir, name)


def _require_out_dir(out_dir: str) -> str:
    if not out_dir:
        raise ValueError("output directory is required")
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"output directory does not exist: {out_dir}")
    return out_dir


def atomic_write_text(out_path: str, content: str) -> None:
    if not out_path:
        raise ValueError("output path is required")
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"output directory does not exist: {out_dir}")
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            _ = handle.write(content)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def fetch_url(url: str, jina_api_key_env: Optional[str], timeout: float) -> str:
    if not url:
        raise ValueError("url is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")

    headers: Dict[str, str] = {}
    if jina_api_key_env:
        api_key = os.environ.get(jina_api_key_env)
        if not api_key:
            raise ValueError(f"missing API key in env var: {jina_api_key_env}")
        headers["Authorization"] = f"Bearer {api_key}"

    target_url = f"https://r.jina.ai/{url}"
    response = requests.get(target_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def add_common_options(parser: argparse.ArgumentParser) -> None:
    _ = parser.add_argument("--jina-api-key-env", default=None)
    _ = parser.add_argument("--timeout", type=float, default=30.0)
    _ = parser.add_argument("--max-chunk-chars", type=int, default=8000)
    _ = parser.add_argument("--concurrency", type=int, default=1)


def cmd_translate_url(args: argparse.Namespace) -> int:
    url = cast(str, args.url)
    out_path = cast(str, args.out)
    jina_api_key_env = cast(Optional[str], args.jina_api_key_env)
    timeout = float(cast(float, args.timeout))
    max_chunk_chars = int(cast(int, args.max_chunk_chars))
    concurrency = int(cast(int, args.concurrency))
    snapdown_to_mermaid = not bool(cast(bool, args.no_snapdown_mermaid))
    if jina_api_key_env:
        api_key = os.environ.get(jina_api_key_env)
        if not api_key:
            raise ValueError(f"missing API key in env var: {jina_api_key_env}")
        os.environ["JINA_API_KEY"] = api_key

    from .pipeline import translate_document

    _ = translate_document(
        source_type="url",
        source_value=url,
        out_path=out_path,
        max_chunk_chars=max_chunk_chars,
        concurrency=concurrency,
        timeout_seconds=timeout,
        snapdown_to_mermaid=snapdown_to_mermaid,
        write_text=atomic_write_text,
    )
    return 0


def cmd_translate_url_batch(args: argparse.Namespace) -> int:
    url_list = cast(Sequence[str], args.url_list)
    out_dir = cast(str, args.out_dir)
    jina_api_key_env = cast(Optional[str], args.jina_api_key_env)
    timeout = float(cast(float, args.timeout))
    max_chunk_chars = int(cast(int, args.max_chunk_chars))
    concurrency = int(cast(int, args.concurrency))
    snapdown_to_mermaid = not bool(cast(bool, args.no_snapdown_mermaid))
    if jina_api_key_env:
        api_key = os.environ.get(jina_api_key_env)
        if not api_key:
            raise ValueError(f"missing API key in env var: {jina_api_key_env}")
        os.environ["JINA_API_KEY"] = api_key

    urls = _collect_url_lists(url_list)
    out_dir = _require_out_dir(out_dir)

    from .pipeline import translate_document

    used_names: Set[str] = set()
    failures: List[str] = []
    for index, url in enumerate(urls, start=1):
        out_path = _build_batch_out_path(out_dir, url, index, used_names)
        try:
            _ = translate_document(
                source_type="url",
                source_value=url,
                out_path=out_path,
                max_chunk_chars=max_chunk_chars,
                concurrency=concurrency,
                timeout_seconds=timeout,
                snapdown_to_mermaid=snapdown_to_mermaid,
                write_text=atomic_write_text,
            )
        except Exception as exc:
            failures.append(f"{url} -> {out_path}: {exc}")

    if failures:
        for line in failures:
            print(f"error: {line}", file=sys.stderr)
        return 1
    return 0


def cmd_translate_md(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    out_path = cast(str, args.out)
    max_chunk_chars = int(cast(int, args.max_chunk_chars))
    concurrency = int(cast(int, args.concurrency))
    title_hint = os.path.basename(input_path)

    from .pipeline import translate_document

    _ = translate_document(
        source_type="file",
        source_value=input_path,
        out_path=out_path,
        max_chunk_chars=max_chunk_chars,
        concurrency=concurrency,
        title_hint=title_hint,
        write_text=atomic_write_text,
    )
    return 0


def cmd_debug_fetch(args: argparse.Namespace) -> int:
    url = cast(str, args.url)
    out_path = cast(str, args.out)
    jina_api_key_env = cast(Optional[str], args.jina_api_key_env)
    timeout = float(cast(float, args.timeout))
    text = fetch_url(url, jina_api_key_env, timeout)
    atomic_write_text(out_path, text)
    return 0


def cmd_debug_chunk(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    max_chunk_chars = int(cast(int, args.max_chunk_chars))
    content = read_text(input_path)
    chunks = build_chunk_plan(content, max_chunk_chars)
    if bool(cast(bool, args.json)):
        print(json.dumps(chunk_plan_payload(chunks), ensure_ascii=True))
    else:
        for chunk in chunks:
            print(f"--- {chunk.chunk_id} ---")
            print(chunk.source_text)
    return 0


def cmd_debug_reconstruct(args: argparse.Namespace) -> int:
    chunks_path = cast(str, args.chunks)
    data = read_text(chunks_path)
    parsed = cast(object, json.loads(data))
    chunks = _parse_chunk_payload(parsed)
    print(reconstruct_from_chunks(chunks), end="")
    return 0


def _parse_chunk_payload(payload: object) -> List[ChunkPlanEntry]:
    if not isinstance(payload, list):
        raise ValueError("chunks must be a JSON array")

    items = cast(List[object], payload)
    if all(isinstance(item, str) for item in items):
        return [
            ChunkPlanEntry(
                chunk_id=f"chunk-{index:04d}",
                source_text=cast(str, item),
                separators=[],
            )
            for index, item in enumerate(items, start=1)
        ]

    chunks: List[ChunkPlanEntry] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("chunks must be a JSON array of objects")
        entry = cast(Dict[str, object], item)
        if (
            "chunk_id" not in entry
            or "source_text" not in entry
            or "separators" not in entry
        ):
            raise ValueError(
                "chunk items must include chunk_id, source_text, and separators"
            )
        chunk_id = entry.get("chunk_id")
        source_text = entry.get("source_text")
        separators_value = entry.get("separators")
        if not isinstance(chunk_id, str):
            raise ValueError("chunk_id must be a string")
        if not isinstance(source_text, str):
            raise ValueError("source_text must be a string")
        if not isinstance(separators_value, list):
            raise ValueError("separators must be a list of strings")
        separators_list = cast(List[object], separators_value)
        for value in separators_list:
            if not isinstance(value, str):
                raise ValueError("separators must be a list of strings")
        separators = cast(List[str], separators_list)
        chunks.append(
            ChunkPlanEntry(
                chunk_id=chunk_id, source_text=source_text, separators=separators
            )
        )

    return chunks


def cmd_debug_protect(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    out_path = cast(str, args.out)
    map_path = cast(str, args.map)
    content = read_text(input_path)
    protected_text, restoration_map = protect(content)
    atomic_write_text(
        map_path, json.dumps(restoration_map, ensure_ascii=True, indent=2)
    )
    atomic_write_text(out_path, protected_text)
    return 0


def cmd_debug_restore(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    map_path = cast(str, args.map)
    out_path = cast(str, args.out)
    content = read_text(input_path)
    map_data = read_text(map_path)
    map_payload = cast(object, json.loads(map_data))
    if not isinstance(map_payload, dict):
        raise PreservationError("map must be a JSON object")
    map_entries = cast(Dict[str, object], map_payload)
    for value in map_entries.values():
        if not isinstance(value, str):
            raise PreservationError("map must be a JSON object of strings")
    restoration_map = cast(Dict[str, str], map_entries)
    try:
        restored = restore(content, restoration_map)
    except PreservationError as exc:
        raise PreservationError(f"restore failed: {exc}") from exc
    atomic_write_text(out_path, restored)
    return 0


def cmd_debug_profile(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    out_path = cast(str, args.out)
    content = read_text(input_path)
    title_hint = os.path.basename(input_path)
    _, markdown = profile_step1(
        content=content,
        source_type="file",
        source_value=input_path,
        title_hint=title_hint,
    )
    atomic_write_text(out_path, markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="translator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    translate_url = subparsers.add_parser("translate-url")
    _ = translate_url.add_argument("--url", required=True)
    _ = translate_url.add_argument("--out", required=True)
    _ = translate_url.add_argument("--no-snapdown-mermaid", action="store_true")
    add_common_options(translate_url)
    translate_url.set_defaults(func=cmd_translate_url)

    translate_url_batch = subparsers.add_parser("translate-url-batch")
    _ = translate_url_batch.add_argument(
        "--url-list", "--url-file", dest="url_list", action="append", required=True
    )
    _ = translate_url_batch.add_argument("--out-dir", required=True)
    _ = translate_url_batch.add_argument("--no-snapdown-mermaid", action="store_true")
    add_common_options(translate_url_batch)
    translate_url_batch.set_defaults(func=cmd_translate_url_batch)

    translate_md = subparsers.add_parser("translate-md")
    _ = translate_md.add_argument("--in", dest="input_path", required=True)
    _ = translate_md.add_argument("--out", required=True)
    add_common_options(translate_md)
    translate_md.set_defaults(func=cmd_translate_md)

    debug_fetch = subparsers.add_parser("debug-fetch")
    _ = debug_fetch.add_argument("--url", required=True)
    _ = debug_fetch.add_argument("--out", required=True)
    _ = debug_fetch.add_argument("--jina-api-key-env", default=None)
    _ = debug_fetch.add_argument("--timeout", type=float, default=30.0)
    debug_fetch.set_defaults(func=cmd_debug_fetch)

    debug_chunk = subparsers.add_parser("debug-chunk")
    _ = debug_chunk.add_argument("--in", dest="input_path", required=True)
    _ = debug_chunk.add_argument("--max-chunk-chars", type=int, default=8000)
    _ = debug_chunk.add_argument("--json", action="store_true")
    debug_chunk.set_defaults(func=cmd_debug_chunk)

    debug_reconstruct = subparsers.add_parser("debug-reconstruct")
    _ = debug_reconstruct.add_argument("--chunks", required=True)
    debug_reconstruct.set_defaults(func=cmd_debug_reconstruct)

    debug_protect = subparsers.add_parser("debug-protect")
    _ = debug_protect.add_argument("--in", dest="input_path", required=True)
    _ = debug_protect.add_argument("--out", required=True)
    _ = debug_protect.add_argument("--map", required=True)
    debug_protect.set_defaults(func=cmd_debug_protect)

    debug_restore = subparsers.add_parser("debug-restore")
    _ = debug_restore.add_argument("--in", dest="input_path", required=True)
    _ = debug_restore.add_argument("--map", required=True)
    _ = debug_restore.add_argument("--out", required=True)
    debug_restore.set_defaults(func=cmd_debug_restore)

    debug_profile = subparsers.add_parser("debug-profile")
    _ = debug_profile.add_argument("--in", dest="input_path", required=True)
    _ = debug_profile.add_argument("--out", required=True)
    debug_profile.set_defaults(func=cmd_debug_profile)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = cast(Callable[[argparse.Namespace], int], args.func)
    return func(args)


def run() -> int:
    try:
        _ = load_dotenv()
        return main()
    except SystemExit as exc:
        if exc.code == 0:
            raise
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
