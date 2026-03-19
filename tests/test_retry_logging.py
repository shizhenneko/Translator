import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.llm_client import KimiClient, _log_llm_retry
from translator.jina_reader_fetcher import _log_jina_retry
from openai import RateLimitError, APIError


def test_llm_retry_logging_enabled(caplog):
    os.environ.pop("TRANSLATOR_RETRY_LOG", None)

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 2
    mock_outcome = Mock()
    mock_exc = RateLimitError(
        message="Rate limit exceeded",
        response=Mock(status_code=429),
        body=None,
    )
    mock_exc.status_code = 429
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome
    mock_next_action = Mock()
    mock_next_action.sleep = 4.5
    mock_retry_state.next_action = mock_next_action

    with caplog.at_level("WARNING"):
        _log_llm_retry(mock_retry_state, "kimi-k2-0905-preview")

    assert len(caplog.records) == 1
    log_message = caplog.records[0].message
    assert "attempt=2" in log_message
    assert "exception=RateLimitError" in log_message
    assert "status=429" in log_message
    assert "sleep=4.50" in log_message
    assert "model=kimi-k2-0905-preview" in log_message

    assert "<<<" not in log_message
    assert ">>>" not in log_message


def test_llm_retry_logging_disabled(caplog):
    os.environ["TRANSLATOR_RETRY_LOG"] = "0"

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 1
    mock_outcome = Mock()
    mock_exc = RateLimitError(
        message="Rate limit exceeded",
        response=Mock(status_code=429),
        body=None,
    )
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome

    with caplog.at_level("WARNING"):
        _log_llm_retry(mock_retry_state, "kimi-k2-0905-preview")

    assert len(caplog.records) == 0

    os.environ.pop("TRANSLATOR_RETRY_LOG", None)


def test_llm_retry_logging_api_error_500(caplog):
    os.environ.pop("TRANSLATOR_RETRY_LOG", None)

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 1
    mock_outcome = Mock()
    mock_response = Mock()
    mock_response.status_code = 503
    mock_exc = APIError(
        message="Service unavailable",
        request=Mock(),
        body=None,
    )
    mock_exc.response = mock_response
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome
    mock_next_action = Mock()
    mock_next_action.sleep = 2.0
    mock_retry_state.next_action = mock_next_action

    with caplog.at_level("WARNING"):
        _log_llm_retry(mock_retry_state, "kimi-k2-0905-preview")

    assert len(caplog.records) == 1
    log_message = caplog.records[0].message
    assert "attempt=1" in log_message
    assert "exception=APIError" in log_message
    assert "status=503" in log_message
    assert "sleep=2.00" in log_message


def test_jina_retry_logging_enabled(caplog):
    os.environ.pop("TRANSLATOR_RETRY_LOG", None)

    import requests

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 3
    mock_outcome = Mock()
    mock_response = Mock()
    mock_response.status_code = 429
    mock_exc = requests.exceptions.HTTPError()
    mock_exc.response = mock_response
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome
    mock_next_action = Mock()
    mock_next_action.sleep = 8.0
    mock_retry_state.next_action = mock_next_action

    with caplog.at_level("WARNING"):
        _log_jina_retry(mock_retry_state)

    assert len(caplog.records) == 1
    log_message = caplog.records[0].message
    assert "attempt=3" in log_message
    assert "exception=HTTPError" in log_message
    assert "status=429" in log_message
    assert "sleep=8.00" in log_message

    assert "<<<" not in log_message
    assert ">>>" not in log_message


def test_jina_retry_logging_disabled(caplog):
    os.environ["TRANSLATOR_RETRY_LOG"] = "0"

    import requests

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 1
    mock_outcome = Mock()
    mock_exc = requests.exceptions.Timeout()
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome

    with caplog.at_level("WARNING"):
        _log_jina_retry(mock_retry_state)

    assert len(caplog.records) == 0

    os.environ.pop("TRANSLATOR_RETRY_LOG", None)


def test_llm_retry_no_privacy_leak(caplog):
    os.environ.pop("TRANSLATOR_RETRY_LOG", None)

    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 1
    mock_outcome = Mock()
    mock_exc = RateLimitError(
        message="Rate limit exceeded for prompt: <<<SECRET_CONTENT>>>",
        response=Mock(status_code=429),
        body=None,
    )
    mock_exc.status_code = 429
    mock_outcome.exception.return_value = mock_exc
    mock_retry_state.outcome = mock_outcome
    mock_next_action = Mock()
    mock_next_action.sleep = 1.0
    mock_retry_state.next_action = mock_next_action

    with caplog.at_level("WARNING"):
        _log_llm_retry(mock_retry_state, "kimi-k2-0905-preview")

    assert len(caplog.records) == 1
    log_message = caplog.records[0].message

    assert "SECRET_CONTENT" not in log_message
    assert "<<<" not in log_message
    assert ">>>" not in log_message
    assert "attempt=1" in log_message
    assert "exception=RateLimitError" in log_message
