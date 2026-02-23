#!/usr/bin/env python3
import argparse
import importlib.util
import os
import sys

script_path = os.path.realpath(__file__)
repo_root = os.path.dirname(os.path.dirname(script_path))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if __package__:
    from .jina_reader_fetcher import JinaReaderConfig, JinaReaderError, fetch_markdown
else:
    module_path = os.path.join(src_path, "translator", "jina_reader_fetcher.py")
    spec = importlib.util.spec_from_file_location(
        "translator_src_jina_reader_fetcher", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load jina_reader_fetcher module")
    _module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_module)
    JinaReaderConfig = getattr(_module, "JinaReaderConfig")
    JinaReaderError = getattr(_module, "JinaReaderError")
    fetch_markdown = getattr(_module, "fetch_markdown")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Markdown with Jina Reader")
    _ = parser.add_argument("url", help="Source URL to fetch")
    _ = parser.add_argument(
        "--min-length",
        type=int,
        default=JinaReaderConfig.min_content_length,
        help="Minimum content length to accept",
    )
    _ = parser.add_argument(
        "--timeout",
        type=int,
        default=JinaReaderConfig.timeout_seconds,
        help="Request timeout in seconds",
    )
    _ = parser.add_argument(
        "--attempts",
        type=int,
        default=JinaReaderConfig.max_attempts,
        help="Maximum retry attempts",
    )

    class Args(argparse.Namespace):
        url: str = ""
        min_length: int = 0
        timeout: int = 0
        attempts: int = 0

    args = parser.parse_args(namespace=Args())

    config = JinaReaderConfig(
        min_content_length=args.min_length,
        timeout_seconds=args.timeout,
        max_attempts=args.attempts,
    )

    try:
        content = fetch_markdown(args.url, config=config)
    except JinaReaderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
