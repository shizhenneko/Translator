from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, TypeVar, cast

import requests


JINA_READER_BASE_URL = "https://r.jina.ai/"
DEFAULT_MIN_CONTENT_LENGTH = 200
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_INITIAL = 1
DEFAULT_BACKOFF_MAX = 20


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
        reraise=True,
    )

    return retrying(do_request)()
