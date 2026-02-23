from __future__ import annotations

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false

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


_PLACEHOLDER_RE = re.compile(r"__([A-Z_]+)_(\d{3})__")


class KimiClient:
    def __init__(
        self,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k2-0905-preview",
        timeout: float = 180.0,
        max_retries: int = 5,
        max_backoff: float = 20.0,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"missing API key in env var: {api_key_env}")
        self._client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
        self._model: str = model
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
