import pytest

from app.router import client_log

pytestmark = pytest.mark.api


class _MockClientLogger:
    def __init__(self):
        self.events = []

    def _emit(self, level, message, fields):
        self.events.append((level, message, fields))

    def error(self, message, **kwargs):
        self._emit("ERROR", message, kwargs.get("extra", {}).get("client_fields", {}))

    def warning(self, message, **kwargs):
        self._emit("WARN", message, kwargs.get("extra", {}).get("client_fields", {}))

    def info(self, message, **kwargs):
        self._emit("INFO", message, kwargs.get("extra", {}).get("client_fields", {}))


def _entry(**overrides):
    base = {
        "level": "ERROR",
        "message": "test message",
        "trace_id": "trace-abc",
        "app": "react",
    }
    base.update(overrides)
    return client_log.ClientLogEntry(**base)


@pytest.fixture
def mock_logger(monkeypatch):
    logger = _MockClientLogger()
    monkeypatch.setattr(client_log, "get_client_logger", lambda: logger)
    return logger


def test_anonymous_error_with_trace_id_written(mock_logger):
    client_log._write_entry(_entry(), user_id=None)
    assert len(mock_logger.events) == 1
    assert mock_logger.events[0][0] == "ERROR"
    assert mock_logger.events[0][1] == "test message"
    assert mock_logger.events[0][2]["trace_id"] == "trace-abc"


def test_anonymous_warn_dropped(mock_logger):
    client_log._write_entry(_entry(level="WARN"), user_id=None)
    assert mock_logger.events == []


def test_anonymous_error_without_trace_id_dropped(mock_logger):
    client_log._write_entry(_entry(trace_id=None), user_id=None)
    assert mock_logger.events == []


def test_logged_in_user_injects_user_id(mock_logger):
    client_log._write_entry(_entry(level="INFO", trace_id="t1"), user_id=42)
    assert len(mock_logger.events) == 1
    assert mock_logger.events[0][0] == "INFO"
    assert mock_logger.events[0][2]["user_id"] == 42


def test_message_and_error_stack_truncated(mock_logger):
    entry = _entry(message="m" * 3000, error_stack="s" * 10000)
    client_log._write_entry(entry, user_id=1)
    fields = mock_logger.events[0][2]
    assert len(mock_logger.events[0][1]) == client_log.MAX_MESSAGE_LENGTH
    assert len(fields["error_stack"]) == client_log.MAX_ERROR_STACK_LENGTH


def test_level_normalization(mock_logger):
    cases = [("error", "ERROR"), ("Warn", "WARN"), ("", "INFO"), (None, "INFO")]
    for level_input, expected_level in cases:
        client_log._write_entry(_entry(level=level_input), user_id=42)
        assert mock_logger.events[-1][0] == expected_level


def test_mixed_anonymous_only_error_written(mock_logger):
    client_log._write_entry(_entry(level="ERROR", trace_id="t1"), user_id=None)
    client_log._write_entry(_entry(level="INFO", trace_id="t2"), user_id=None)
    client_log._write_entry(_entry(level="WARN", trace_id="t3"), user_id=None)

    assert len(mock_logger.events) == 1
    assert mock_logger.events[0][0] == "ERROR"


def test_blank_string_fields_excluded(mock_logger):
    entry = _entry(
        app="react",
        url="   ",
        user_agent="",
        method="POST",
        path=None,
    )
    client_log._write_entry(entry, user_id=42)

    fields = mock_logger.events[0][2]
    assert fields["app"] == "react"
    assert fields["method"] == "POST"
    assert "url" not in fields
    assert "user_agent" not in fields
    assert "path" not in fields


def test_numeric_fields_nullable(mock_logger):
    entry = _entry(status=500, duration=1203.5)
    client_log._write_entry(entry, user_id=42)

    fields = mock_logger.events[0][2]
    assert fields["status"] == 500
    assert fields["duration"] == 1203.5
    assert "metric_value" not in fields
