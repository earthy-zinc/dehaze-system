"""调试工具库：SSE 流式请求客户端。

解决后端联调时手搓 SSE 请求的常见坑：
- POST /api/v1/ai/conversations/{id}/messages 必需 Idempotency-Key 头（uuid hex），缺失返回 A0400
- 响应是标准 SSE 分行（id:/event:/data:），非 JSON，utils/api.py 的 request() 无法处理

SSE 事件块格式（与 dehaze-python/app/infrastructure/sse/sse_emitter_manager.py 对齐）：
    id: {eventId}
    event: {eventType}
    data: {json}
    空行分隔
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

import httpx

from . import api, auth, config


@dataclass
class SseStreamResult:
    """SSE 流式响应解析结果。"""

    # (event_name, payload_dict) 有序列表，已跳过 ping
    events: list[tuple[str, dict]] = field(default_factory=list)
    # content_block.delta 中 delta.type=text_delta 的 delta.text 全量拼接
    text: str = ""
    # thought 事件的 thought 字段拼接（无思考过程时为空串）
    thought: str = ""


def stream_request(
    method: str,
    path: str,
    backend: str = "python",
    json_body: dict | None = None,
) -> SseStreamResult:
    """发起带会话凭证与幂等头的 SSE 流式请求并完整解析。

    - 复用 utils/api.py 的 session 状态（X-Session-Id 自动登录/复用）
    - 自动生成 Idempotency-Key（uuid4().hex）
    - 超时 120s；流式中途的 error 事件不抛异常，记录在 events 里由调用方判断
    """
    session_id = api._sessions.get(backend)
    if not session_id:
        session_id = auth.login(backend=backend)

    headers = {
        "X-Session-Id": session_id,
        "Idempotency-Key": uuid.uuid4().hex,
    }
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    backend_cfg = config.get_backend(backend)
    with httpx.Client(base_url=backend_cfg.base_url, timeout=120) as client:
        resp = client.request(method, path, headers=headers, json=json_body)
        resp.raise_for_status()

    result = SseStreamResult()
    event_name = None
    data_lines: list[str] = []
    for line in resp.text.splitlines():
        if line == "":
            # 空行结束当前事件块
            if event_name and data_lines:
                _handle_event(result, event_name, "\n".join(data_lines))
            event_name = None
            data_lines = []
        elif line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # id: 行仅用于断点续传，解析时无需使用
    # 末尾可能没有空行结尾
    if event_name and data_lines:
        _handle_event(result, event_name, "\n".join(data_lines))
    return result


def _handle_event(result: SseStreamResult, event_name: str, data_str: str) -> None:
    if event_name == "ping":
        return
    try:
        payload = json.loads(data_str)
    except json.JSONDecodeError:
        payload = {"raw": data_str}
    if not isinstance(payload, dict):
        payload = {"data": payload}
    result.events.append((event_name, payload))
    if event_name == "content_block.delta":
        delta = payload.get("delta") or {}
        if delta.get("type") == "text_delta":
            result.text += delta.get("text") or ""
    elif event_name == "thought":
        result.thought += payload.get("thought") or ""
