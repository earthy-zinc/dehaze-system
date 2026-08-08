"""调试工具库：HTTP 客户端（httpx）。

对齐 dehaze-sdk-js/src/utils/request.ts：
- 按 backend 维护单例 httpx.Client
- 自动注入 X-Session-Id 请求头
- 提供 get / post / put / patch / delete
- 响应体自动解析 JSON，失败时抛出 DehazeApiError（含 traceId）
"""
from __future__ import annotations

import httpx

from . import config, auth


class DehazeApiError(Exception):
    """后端 API 业务错误。"""
    def __init__(self, code: str, msg: str, trace_id: str | None = None):
        super().__init__(f"[{code}] {msg} (traceId={trace_id})")
        self.code = code
        self.msg = msg
        self.trace_id = trace_id


# 按 backend 维护 Client 单例
_clients: dict[str, httpx.Client] = {}
# 按 backend 维护当前 session_id
_sessions: dict[str, str | None] = {}


def _get_client(backend: str) -> httpx.Client:
    if backend not in _clients:
        backend_cfg = config.get_backend(backend)
        _clients[backend] = httpx.Client(base_url=backend_cfg.base_url, timeout=30)
    return _clients[backend]


def set_session(backend: str, session_id: str | None) -> None:
    """设置指定后端的当前 session_id。"""
    _sessions[backend] = session_id


def clear_sessions() -> None:
    """清除所有 session 状态。"""
    _sessions.clear()


def request(method: str, path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    """统一请求入口。"""
    client = _get_client(backend)
    headers = kwargs.pop("headers", {}) or {}

    # 自动注入 X-Session-Id
    session_id = _sessions.get(backend)
    if session_id:
        headers["X-Session-Id"] = session_id

    # 未指定 backend 的 session 时尝试自动登录
    if not session_id and not path.startswith("/api/v1/auth/"):
        sid = auth.login(backend=backend)
        headers["X-Session-Id"] = sid

    resp = client.request(method, path, headers=headers, **kwargs)
    resp.raise_for_status()

    data = resp.json()
    code = data.get("code")
    if code != config.SUCCESS_CODE:
        raise DehazeApiError(code, data.get("msg", ""), data.get("traceId"))
    return data


def get(path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    return request("GET", path, backend=backend, **kwargs)


def post(path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    return request("POST", path, backend=backend, **kwargs)


def put(path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    return request("PUT", path, backend=backend, **kwargs)


def patch(path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    return request("PATCH", path, backend=backend, **kwargs)


def delete(path: str, backend: str = config.DEFAULT_BACKEND, **kwargs) -> dict:
    return request("DELETE", path, backend=backend, **kwargs)


def close() -> None:
    """关闭所有 HTTP 客户端。"""
    for client in _clients.values():
        client.close()
    _clients.clear()
    _sessions.clear()
