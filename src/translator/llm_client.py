from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false

import logging
import os
import re
from typing import Dict, List, Optional, cast

from openai import APIError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from tenacity import (  # type: ignore[import-not-found]
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


_PLACEHOLDER_RE = re.compile(r"(?<![_A-Za-z0-9])__([A-Z][A-Z_]*)_(\d{3})__")
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$",
    re.DOTALL,
)
_DEFAULT_MODEL = "kimi-k2-0905-preview"
_DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
_MODEL_ENV = "MOONSHOT_MODEL"
_BASE_URL_ENV = "MOONSHOT_BASE_URL"

logger = logging.getLogger(__name__)


def _log_llm_retry(retry_state, model: str) -> None:
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
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status = getattr(response, "status_code", None)
    logger.warning(
        "LLM retry attempt=%d exception=%s status=%s sleep=%.2fs model=%s",
        attempt,
        exc_class,
        status,
        sleep_sec,
        model,
    )


class KimiClient:
    def __init__(
        self,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 180.0,
        max_retries: int = 5,
        max_backoff: float = 20.0,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"missing API key in env var: {api_key_env}")
        env_base_url = os.environ.get(_BASE_URL_ENV)
        resolved_base_url = (
            base_url
            or (env_base_url.strip() if env_base_url else None)
            or _DEFAULT_BASE_URL
        )
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=resolved_base_url)
        env_model = os.environ.get(_MODEL_ENV)
        resolved_model = (
            model or (env_model.strip() if env_model else None) or _DEFAULT_MODEL
        )
        self._model: str = resolved_model
        self._timeout: float = timeout
        self._max_retries: int = max_retries
        self._max_backoff: float = max_backoff

    def chat_completion(
        self,
        messages: List[ChatCompletionMessageParam],
        json_mode: bool = False,
        timeout: Optional[float] = None,
        preservation_map: Optional[Dict[str, object]] = None,
        expected_placeholders: Optional[List[str]] = None,
    ) -> str:
        request_timeout = self._timeout if timeout is None else timeout
        response = self._request_with_retry(messages, json_mode, request_timeout)
        if not response.choices:
            raise RuntimeError("chat completion returned no choices")

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason != "stop":
            raise RuntimeError(
                f"chat completion truncated (finish_reason={finish_reason})"
            )

        message = choice.message
        content = message.content
        if not isinstance(content, str) or not content:
            raise RuntimeError("chat completion returned empty content")

        if json_mode:
            content = self._strip_code_fences(content)

        if preservation_map is not None:
            self._validate_preservation_map(content, preservation_map)
        elif expected_placeholders is not None:
            self._validate_expected_placeholders(content, expected_placeholders)

        return content

    def _request_with_retry(
        self,
        messages: List[ChatCompletionMessageParam],
        json_mode: bool,
        timeout: float,
    ) -> ChatCompletion:
        retrying = Retrying(
            retry=retry_if_exception(self._is_retryable_error),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_random_exponential(multiplier=1, max=self._max_backoff),
            before_sleep=lambda retry_state: _log_llm_retry(retry_state, self._model),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                if json_mode:
                    return self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        timeout=timeout,
                        response_format={"type": "json_object"},
                    )
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    timeout=timeout,
                )
        raise RuntimeError("retrying chat completion exhausted unexpectedly")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        match = _CODE_FENCE_RE.match(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _is_retryable_error(exc: BaseException) -> bool:
        if isinstance(exc, RateLimitError):
            return True
        if isinstance(exc, APIError):
            status = getattr(exc, "status_code", None)
            if status is None:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
            if status in (429,) or (isinstance(status, int) and status >= 500):
                return True
        return False

    @staticmethod
    def _validate_expected_placeholders(
        content: str, expected_placeholders: List[str]
    ) -> None:
        placeholders = list(expected_placeholders)
        placeholder_set = set(placeholders)
        for placeholder in placeholders:
            count = content.count(placeholder)
            if count == 0:
                raise RuntimeError(f"placeholder missing: {placeholder}")
            if count > 1:
                raise RuntimeError(
                    f"placeholder duplicated: {placeholder} (count={count})"
                )

        for match in _PLACEHOLDER_RE.finditer(content):
            placeholder = match.group(0)
            if placeholder not in placeholder_set:
                raise RuntimeError(f"unknown placeholder found: {placeholder}")

    @staticmethod
    def _validate_preservation_map(content: str, preservation_map: object) -> None:
        if not isinstance(preservation_map, dict):
            raise ValueError("preservation map must be a dict")
        preservation_map_typed = cast(Dict[str, object], preservation_map)
        placeholders_obj = preservation_map_typed.get("placeholders")
        if not isinstance(placeholders_obj, list):
            raise ValueError("preservation map missing placeholders list")
        placeholders: List[str] = []
        entries = cast(List[object], placeholders_obj)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("preservation map placeholder entry must be a dict")
            entry_dict = cast(Dict[str, object], entry)
            placeholder = entry_dict.get("placeholder")
            if not isinstance(placeholder, str):
                raise ValueError("preservation map placeholder must be a string")
            placeholders.append(placeholder)
        KimiClient._validate_expected_placeholders(content, placeholders)
