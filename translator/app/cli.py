import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, cast

import requests
from dotenv import load_dotenv
from ..core.chunking import (
    ChunkPlanEntry,
    build_chunk_plan,
    chunk_plan_payload,
    reconstruct_from_chunks,
)
from ..markdown.autofix import MarkdownAutofixOptions, autofix_markdown
from ..markdown.lint import MarkdownLintOptions, format_issue_report, lint_markdown
from ..markdown.sanitize import sanitize_markdown_input
from ..io.fs_utils import atomic_write_text, read_text
from ..core.preservation import PreservationError, protect, restore
from ..core.step1_profile import profile as profile_step1
from ..services.translation_runner import (
    TranslationOptions,
    collect_url_lists,
    normalize_urls,
    require_out_dir,
    translate_url_to_path,
    translate_urls_batch,
)


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
    _ = parser.add_argument("--max-chunk-chars", type=int, default=5000)
    _ = parser.add_argument("--concurrency", type=int, default=2)
    _ = parser.add_argument(
        "--prompt-outline-mode",
        choices=["headings", "full"],
        default="headings",
        help="Outline rendering mode: 'headings' (slim) or 'full' (legacy with summaries)",
    )
    _ = parser.add_argument(
        "--prompt-glossary-mode",
        choices=["filtered", "full"],
        default="filtered",
        help="Glossary mode: 'filtered' (chunk-relevant terms only) or 'full' (all terms)",
    )
    _ = parser.add_argument(
        "--output-format",
        choices=["readable", "analysis"],
        default="readable",
        help="Output rendering mode: 'readable' (default) or 'analysis'",
    )


def _translation_options_from_args(args: argparse.Namespace) -> TranslationOptions:
    return TranslationOptions(
        jina_api_key_env=cast(Optional[str], getattr(args, "jina_api_key_env", None)),
        timeout=float(cast(float, getattr(args, "timeout", 30.0))),
        max_chunk_chars=int(cast(int, getattr(args, "max_chunk_chars", 5000))),
        concurrency=int(cast(int, getattr(args, "concurrency", 2))),
        snapdown_to_mermaid=not bool(
            cast(bool, getattr(args, "no_snapdown_mermaid", False))
        ),
        prompt_outline_mode=cast(
            str, getattr(args, "prompt_outline_mode", "headings")
        ),
        prompt_glossary_mode=cast(
            str, getattr(args, "prompt_glossary_mode", "filtered")
        ),
        output_format=cast(str, getattr(args, "output_format", "readable")),
    )


def cmd_translate_url(args: argparse.Namespace) -> int:
    url_values = cast(Sequence[str], args.url)
    urls = normalize_urls(url_values)
    if len(urls) != 1:
        raise ValueError("single-document translation requires exactly one --url value")
    url = urls[0]
    out_path = cast(str, args.out)
    _ = translate_url_to_path(
        url=url,
        out_path=out_path,
        options=_translation_options_from_args(args),
        write_text=atomic_write_text,
    )
    return 0


def cmd_translate_url_batch(args: argparse.Namespace) -> int:
    out_dir = require_out_dir(cast(str, args.out_dir))
    urls = _resolve_batch_urls(args)
    results = translate_urls_batch(
        urls=urls,
        out_dir=out_dir,
        options=_translation_options_from_args(args),
        write_text=atomic_write_text,
    )
    failures = [
        f"{result.url} -> {result.out_path}: {result.error}"
        for result in results
        if not result.success
    ]
    success_count = sum(1 for result in results if result.success)

    if failures:
        for line in failures:
            print(f"error: {line}", file=sys.stderr)
    return 0 if success_count > 0 else 1


def cmd_translate_md(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    out_path = cast(str, args.out)
    max_chunk_chars = int(cast(int, args.max_chunk_chars))
    concurrency = int(cast(int, args.concurrency))
    prompt_outline_mode = cast(str, args.prompt_outline_mode)
    prompt_glossary_mode = cast(str, args.prompt_glossary_mode)
    output_format = cast(str, args.output_format)
    title_hint = os.path.basename(input_path)

    from ..core.pipeline import translate_document

    _ = translate_document(
        source_type="file",
        source_value=input_path,
        out_path=out_path,
        max_chunk_chars=max_chunk_chars,
        concurrency=concurrency,
        title_hint=title_hint,
        prompt_outline_mode=prompt_outline_mode,
        prompt_glossary_mode=prompt_glossary_mode,
        output_format=output_format,
        write_text=atomic_write_text,
    )
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    url_values = cast(Sequence[str], getattr(args, "url", []) or [])
    input_path = cast(Optional[str], getattr(args, "input_path", None))
    url_list = cast(Optional[Sequence[str]], getattr(args, "url_list", None))
    urls = [value.strip() for value in url_values if value.strip()]
    if url_list or len(urls) > 1:
        out_dir = cast(Optional[str], getattr(args, "out_dir", None))
        out_path = cast(Optional[str], getattr(args, "out", None))
        if out_path and not out_dir:
            args.out_dir = out_path
        if not cast(Optional[str], getattr(args, "out_dir", None)):
            raise ValueError("batch translation requires --out-dir (or use --out as the batch directory)")
        return cmd_translate_url_batch(args)
    out_path = cast(Optional[str], getattr(args, "out", None))
    if not out_path:
        raise ValueError("--out is required for single-document translation")
    if bool(urls) == bool(input_path):
        raise ValueError("exactly one of --url or --in is required unless batch URL mode is used")
    if urls:
        args.url = urls
        return cmd_translate_url(args)
    args.input_path = cast(str, input_path)
    return cmd_translate_md(args)


def _resolve_batch_urls(args: argparse.Namespace) -> List[str]:
    urls: List[str] = []
    inline_values = cast(Sequence[str], getattr(args, "url", []) or [])
    if inline_values:
        urls.extend(normalize_urls(inline_values))
    url_list = cast(Sequence[str], getattr(args, "url_list", []) or [])
    if url_list:
        urls.extend(collect_url_lists(url_list))
    if not urls:
        raise ValueError("no URLs provided")
    return urls


def cmd_serve(args: argparse.Namespace) -> int:
    from ..web.app import serve_app

    serve_app(
        host=cast(str, args.host),
        port=int(cast(int, args.port)),
        job_base_dir=cast(Optional[str], getattr(args, "job_dir", None)),
        max_workers=int(cast(int, getattr(args, "workers", 2))),
    )
    return 0


def cmd_lint_md(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    fix = bool(cast(bool, args.fix))
    out_path = cast(Optional[str], args.out)
    in_place = bool(cast(bool, args.in_place))
    strict_renderer = cast(str, args.strict_renderer).lower() != "off"
    max_safe_list_depth = int(cast(int, args.max_safe_list_depth))

    try:
        if max_safe_list_depth <= 0:
            raise ValueError("--max-safe-list-depth must be positive")
        lint_options = MarkdownLintOptions(
            strict_renderer=strict_renderer, max_safe_list_depth=max_safe_list_depth
        )
        autofix_options = MarkdownAutofixOptions(
            strict_renderer=strict_renderer, max_safe_list_depth=max_safe_list_depth
        )

        content = read_text(input_path)

        if not fix:
            issues = lint_markdown(content, options=lint_options)
            if issues:
                print(format_issue_report(issues), file=sys.stderr)
                return 1
            return 0

        if out_path and in_place:
            raise ValueError("--out and --in-place cannot be used together")
        if not out_path and not in_place:
            raise ValueError("--fix requires --out or --in-place")

        sanitized = sanitize_markdown_input(content, aggressive=True)
        fixed = autofix_markdown(sanitized, options=autofix_options)
        issues = lint_markdown(fixed, options=lint_options)
        if issues:
            print(format_issue_report(issues), file=sys.stderr)
            return 1

        target_path = input_path if in_place else cast(str, out_path)
        atomic_write_text(target_path, fixed)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_sanitize_md(args: argparse.Namespace) -> int:
    input_path = cast(str, args.input_path)
    out_path = cast(Optional[str], args.out)
    in_place = bool(cast(bool, args.in_place))
    try:
        if out_path and in_place:
            raise ValueError("--out and --in-place cannot be used together")
        if not out_path and not in_place:
            raise ValueError("--out or --in-place is required")
        content = read_text(input_path)
        sanitized = sanitize_markdown_input(content, aggressive=True)
        target_path = input_path if in_place else cast(str, out_path)
        atomic_write_text(target_path, sanitized)
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


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

    serve = subparsers.add_parser("serve")
    _ = serve.add_argument("--host", default="127.0.0.1")
    _ = serve.add_argument("--port", type=int, default=10001)
    _ = serve.add_argument("--job-dir")
    _ = serve.add_argument("--workers", type=int, default=2)
    serve.set_defaults(func=cmd_serve)

    translate = subparsers.add_parser("translate")
    source_group = translate.add_mutually_exclusive_group(required=True)
    _ = source_group.add_argument("--url", action="append")
    _ = source_group.add_argument("--in", dest="input_path")
    _ = source_group.add_argument("--url-list", "--url-file", dest="url_list", action="append")
    _ = translate.add_argument("--out")
    _ = translate.add_argument("--out-dir")
    _ = translate.add_argument("--no-snapdown-mermaid", action="store_true")
    add_common_options(translate)
    translate.set_defaults(func=cmd_translate)

    translate_url = subparsers.add_parser("translate-url")
    _ = translate_url.add_argument("--url", action="append", required=True)
    _ = translate_url.add_argument("--out", required=True)
    _ = translate_url.add_argument("--no-snapdown-mermaid", action="store_true")
    add_common_options(translate_url)
    translate_url.set_defaults(func=cmd_translate_url)

    translate_url_batch = subparsers.add_parser("translate-url-batch")
    _ = translate_url_batch.add_argument("--url", action="append", default=[])
    _ = translate_url_batch.add_argument(
        "--url-list", "--url-file", dest="url_list", action="append"
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

    lint_md = subparsers.add_parser("lint-md")
    _ = lint_md.add_argument("--in", dest="input_path", required=True)
    _ = lint_md.add_argument("--fix", action="store_true")
    _ = lint_md.add_argument("--out")
    _ = lint_md.add_argument("--in-place", action="store_true")
    _ = lint_md.add_argument(
        "--strict-renderer",
        choices=["on", "off"],
        default="on",
        help="Enable markdown-it renderer safety checks",
    )
    _ = lint_md.add_argument("--max-safe-list-depth", type=int, default=1)
    lint_md.set_defaults(func=cmd_lint_md)

    sanitize_md = subparsers.add_parser("sanitize-md")
    _ = sanitize_md.add_argument("--in", dest="input_path", required=True)
    _ = sanitize_md.add_argument("--out")
    _ = sanitize_md.add_argument("--in-place", action="store_true")
    sanitize_md.set_defaults(func=cmd_sanitize_md)

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
