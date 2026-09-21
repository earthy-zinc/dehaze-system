from unittest.mock import MagicMock

from starlette.requests import Request

from app.middleware.api_key_auth import ApiKeyAuthMiddleware


async def _noop_app(scope, receive, send):  # pragma: no cover
    raise AssertionError("中间件测试仅测 _audit_401，不调用 ASGI app")


def _make_mw():
    rc = MagicMock()
    mw = ApiKeyAuthMiddleware(app=_noop_app, record_call=rc)
    return mw, rc


def _request(path: str, headers: dict | None = None) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "query_string": b"",
            "headers": raw_headers,
            "client": ("203.0.113.9", 0),
            "server": ("test", 80),
        }
    )


class TestAudit401:
    def test_compat_endpoint_openai_protocol(self):
        mw, rc = _make_mw()
        mw._audit_401(
            _request("/api/v1/chat/completions", {"authorization": "Bearer dhak_ab3xyz..."}),
            "dhak_ab3xyz9f8g7h6",
            "API Key 无效或已禁用",
        )
        kwargs = rc.call_args.kwargs
        assert rc.call_count == 1
        assert kwargs["user_id"] is None
        assert kwargs["key_id"] is None
        assert kwargs["key_prefix"] == "dhak_ab3"
        assert kwargs["endpoint"] == "completions"
        assert kwargs["protocol"] == "openai"
        assert kwargs["status_code"] == 401
        assert kwargs["client_ip"] == "203.0.113.9"

    def test_compat_endpoint_claude_protocol(self):
        mw, rc = _make_mw()
        mw._audit_401(
            _request("/api/v1/messages", {"x-api-key": "dhak_ab3xyz..."}),
            "dhak_ab3xyz9f8g7h6",
            "API Key 已过期",
        )
        kwargs = rc.call_args.kwargs
        assert rc.call_count == 1
        assert kwargs["protocol"] == "claude"
        assert kwargs["endpoint"] == "messages"

    def test_non_compat_endpoint_not_recorded(self):
        mw, rc = _make_mw()
        mw._audit_401(
            _request("/api/v1/system/users/list"),
            "dhak_ab3xyz9f8g7h6",
            "API Key 无效或已禁用",
        )
        assert rc.call_count == 0
