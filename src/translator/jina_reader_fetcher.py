from __future__ import annotations

import html
import importlib
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, TypeVar, cast

import requests


JINA_READER_BASE_URL = "https://r.jina.ai/"
DEFAULT_MIN_CONTENT_LENGTH = 200
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_INITIAL = 1
DEFAULT_BACKOFF_MAX = 20

logger = logging.getLogger(__name__)


def _log_jina_retry(retry_state) -> None:
    if os.environ.get("TRANSLATOR_RETRY_LOG", "1") == "0":
        return
    attempt = retry_state.attempt_number
    outcome = retry_state.outcome
    if outcome is None:
        return
    exc = outcome.exception()
    if exc is None:
        return
    exc_class = type(exc).__name__
    sleep_sec = 0.0
    if retry_state.next_action is not None:
        sleep_sec = retry_state.next_action.sleep
    status = None
    if isinstance(exc, requests.RequestException):
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    logger.warning(
        "Jina Reader retry attempt=%d exception=%s status=%s sleep=%.2fs",
        attempt,
        exc_class,
        status,
        sleep_sec,
    )


class JinaReaderError(RuntimeError):
    pass


class JinaReaderTransientError(JinaReaderError):
    pass


T = TypeVar("T")


class TenacityModule(Protocol):
    def retry(
        self,
        *,
        retry: object,
        wait: object,
        stop: object,
        reraise: bool,
        before_sleep: Optional[object] = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]: ...

    def retry_if_exception_type(self, exception_types: object) -> object: ...

    def wait_exponential_jitter(self, *, initial: float, max: float) -> object: ...

    def stop_after_attempt(self, max_attempt_number: int) -> object: ...


@dataclass(frozen=True)
class JinaReaderConfig:
    min_content_length: int = DEFAULT_MIN_CONTENT_LENGTH
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_initial: int = DEFAULT_BACKOFF_INITIAL
    backoff_max: int = DEFAULT_BACKOFF_MAX


@dataclass(frozen=True)
class SnapdownBlock:
    language: str
    content: str
    heading: Optional[str] = None


def _build_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "X-Return-Format": "markdown",
    }
    api_key = os.getenv("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_content(payload: Dict[str, object]) -> Optional[str]:
    content = payload.get("content")
    if isinstance(content, str):
        return content
    data = payload.get("data")
    if isinstance(data, dict):
        data_map = cast(Dict[str, object], data)
        content = data_map.get("content")
        if isinstance(content, str):
            return content
    return None


_SNAPDOWN_SCRIPT_RE = re.compile(
    r"<script\b[^>]*\btype\s*=\s*['\"]application/snapdown(?P<json>\+json)?['\"][^>]*>(?P<content>.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)

_HEADING_RE = re.compile(
    r"<h(?P<level>[1-6])\b[^>]*>(?P<content>.*?)</h(?P=level)>",
    flags=re.IGNORECASE | re.DOTALL,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_heading(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized.strip()


def _strip_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub("", text)


def extract_snapdown_blocks_from_html(html_text: str) -> List[SnapdownBlock]:
    headings: List[tuple[int, str]] = []
    for match in _HEADING_RE.finditer(html_text):
        content = match.group("content")
        if content is None:
            continue
        text = _normalize_heading(html.unescape(_strip_html_tags(content)))
        if not text:
            continue
        headings.append((match.start(), text))

    blocks: List[SnapdownBlock] = []
    for match in _SNAPDOWN_SCRIPT_RE.finditer(html_text):
        if match.group("json"):
            continue
        raw_content = match.group("content")
        if raw_content is None:
            continue
        content = html.unescape(raw_content).strip()
        if not content:
            continue
        heading = None
        for pos, text in reversed(headings):
            if pos < match.start():
                heading = text
                break
        blocks.append(
            SnapdownBlock(language="snapdown", content=content, heading=heading)
        )
    return blocks


def _build_fence(content: str) -> str:
    runs = cast(List[str], re.findall(r"`+", content))
    max_len = max((len(run) for run in runs), default=0)
    fence_len = max(3, max_len + 1)
    return "`" * fence_len


def _render_snapdown_section(blocks: Sequence[SnapdownBlock]) -> str:
    if not blocks:
        return ""
    lines: List[str] = ["## Snapdown Diagrams (extracted)", ""]
    for block in blocks:
        fence = _build_fence(block.content)
        lines.append(f"{fence}{block.language}")
        lines.append(block.content)
        lines.append(fence)
        lines.append("")
    if lines[-1] == "":
        _ = lines.pop()
    return "\n".join(lines)


def _response_error_message(
    response: requests.Response, payload: Optional[Dict[str, object]]
) -> str:
    if payload is None:
        return f"Unexpected HTTP {response.status_code} with non-JSON body"
    code = payload.get("code")
    status = payload.get("status")
    message = payload.get("message")
    return f"Unexpected response code={code} status={status} message={message}"


def _is_transient_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def fetch_snapdown_blocks(
    url: str, config: Optional[JinaReaderConfig] = None
) -> List[SnapdownBlock]:
    if not url.strip():
        raise JinaReaderError("URL must be a non-empty string")
    config = config or JinaReaderConfig()
    clean_url = url.strip()
    headers = {"User-Agent": "translator/1.0"}
    try:
        response = requests.get(
            clean_url, headers=headers, timeout=config.timeout_seconds
        )
        response.raise_for_status()
    except requests.RequestException:
        return []
    return extract_snapdown_blocks_from_html(response.text)


def append_snapdown_blocks(markdown: str, blocks: Sequence[SnapdownBlock]) -> str:
    if not blocks:
        return markdown
    section = _render_snapdown_section(blocks)
    if not section:
        return markdown
    normalized = markdown.rstrip("\n")
    return f"{normalized}\n\n{section}\n"


def insert_snapdown_blocks(markdown: str, blocks: Sequence[SnapdownBlock]) -> str:
    if not blocks:
        return markdown
    lines = markdown.splitlines()
    used = 0
    cursor = 0
    blocks_with_heading = [block for block in blocks if block.heading]

    for block in blocks_with_heading:
        heading = _normalize_heading(cast(str, block.heading))
        for index in range(cursor, len(lines)):
            line = lines[index]
            if not line.lstrip().startswith("#"):
                continue
            title = _normalize_heading(line.lstrip("# "))
            if title != heading:
                continue
            insert_at = index + 1
            while insert_at < len(lines) and lines[insert_at].strip() == "":
                insert_at += 1
            fence = _build_fence(block.content)
            block_lines = [
                f"{fence}{block.language}",
                block.content,
                fence,
                "",
            ]
            lines[insert_at:insert_at] = block_lines
            cursor = insert_at + len(block_lines)
            used += 1
            break

    remaining = list(blocks)[used:]
    if not remaining:
        return "\n".join(lines)
    return append_snapdown_blocks("\n".join(lines), remaining)


def fetch_markdown(url: str, config: Optional[JinaReaderConfig] = None) -> str:
    if not url.strip():
        raise JinaReaderError("URL must be a non-empty string")

    config = config or JinaReaderConfig()
    headers = _build_headers()
    clean_url = url.strip()

    def do_request() -> str:
        if "#" in clean_url:
            response = requests.post(
                JINA_READER_BASE_URL,
                data={"url": clean_url},
                headers=headers,
                timeout=config.timeout_seconds,
            )
        else:
            response = requests.get(
                f"{JINA_READER_BASE_URL}{clean_url}",
                headers=headers,
                timeout=config.timeout_seconds,
            )

        if _is_transient_status(response.status_code):
            raise JinaReaderTransientError(
                f"Transient HTTP {response.status_code} from Jina Reader"
            )

        payload: Optional[Dict[str, object]]
        try:
            payload_obj = cast(object, response.json())
        except ValueError:
            payload_obj = None

        if isinstance(payload_obj, dict):
            payload = cast(Dict[str, object], payload_obj)
        else:
            payload = None

        if response.status_code != 200:
            raise JinaReaderError(_response_error_message(response, payload))

        if payload is None:
            raise JinaReaderError("Expected JSON response from Jina Reader")

        if payload.get("code") != 200:
            raise JinaReaderError(_response_error_message(response, payload))

        content = _extract_content(payload)
        if not content:
            raise JinaReaderError("Missing content in Jina Reader response")

        if len(content) < config.min_content_length:
            raise JinaReaderError(
                f"Content too short ({len(content)} < {config.min_content_length})"
            )

        return content

    try:
        tenacity = cast(
            TenacityModule,
            cast(object, importlib.import_module("tenacity")),
        )
    except ModuleNotFoundError as exc:
        raise JinaReaderError(
            "tenacity is required for retry logic; install it before running"
        ) from exc

    retrying = tenacity.retry(
        retry=tenacity.retry_if_exception_type(
            (JinaReaderTransientError, requests.RequestException)
        ),
        wait=tenacity.wait_exponential_jitter(
            initial=config.backoff_initial, max=config.backoff_max
        ),
        stop=tenacity.stop_after_attempt(config.max_attempts),
        before_sleep=_log_jina_retry,
        reraise=True,
    )

    return retrying(do_request)()
