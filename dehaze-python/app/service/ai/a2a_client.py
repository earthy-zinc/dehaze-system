"""A2A 协议客户端（A2AClient）

平台作为 A2A 客户端调用外部 A2A Agent 作为子 Agent：
- fetch_agent_card：注册端点时拉取并校验 Agent Card（SSRF 防护）
- message_send：message/send 异步调用，轮询 tasks/get
- message_stream：message/stream，解析 SSE 事件
- 凭证 AES 加密存储（复用 aes_cipher），运行时解密按声明方案注入请求头
- SSRF 防护：base_url / agent_card_url 仅 https 且禁内网
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.models.entity.sys_ai_agent_endpoint import SysAiAgentEndpoint
from app.service.ai.a2a_protocol import JsonRpcRequest, Task
from app.utils.ssrf import is_safe_url

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """A2A 客户端调用错误"""


class A2AClient:
    """A2A 协议客户端（单例）"""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    # ── Agent Card 拉取 ────────────────────────────────────────

    async def fetch_agent_card(self, agent_card_url: str) -> dict[str, Any]:
        """拉取并校验外部 Agent Card（SSRF 防护）。"""
        if not await is_safe_url(agent_card_url):
            raise A2AClientError("Agent Card URL 仅支持 https 且禁止内网地址")
        try:
            resp = await self._client.get(agent_card_url)
            resp.raise_for_status()
            card = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise A2AClientError(f"拉取 Agent Card 失败: {e}") from e
        if not isinstance(card, dict) or "name" not in card:
            raise A2AClientError("Agent Card 格式非法")
        return card

    # ── 请求构建 ───────────────────────────────────────────────

    @staticmethod
    def _build_headers(endpoint: SysAiAgentEndpoint) -> dict:
        """按端点声明方案注入认证头。"""
        from app.infrastructure.crypto.aes_cipher import decrypt

        headers = {"Content-Type": "application/json"}
        credential = endpoint.credential or ""
        if not credential:
            return headers
        try:
            secret = decrypt(credential)
        except Exception:
            logger.warning("端点 %s 凭证解密失败，跳过认证头", endpoint.id)
            return headers
        if endpoint.auth_type == "apiKey":
            headers["X-API-Key"] = secret
        elif endpoint.auth_type == "http":
            headers["Authorization"] = f"Bearer {secret}"
        # oauth2 / openIdConnect / mutualTLS：凭证为 token，按 http 头注入
        else:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    @staticmethod
    def _endpoint_url(endpoint: SysAiAgentEndpoint) -> str:
        """拼接触发端点（a2a 端点地址）。"""
        return endpoint.base_url

    # ── RPC 调用 ───────────────────────────────────────────────

    async def _rpc(self, endpoint: SysAiAgentEndpoint, method: str, params: dict) -> dict:
        """发送 JSON-RPC 请求并解析响应。"""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        url = self._endpoint_url(endpoint)
        # 运行期复检一次 SSRF（短缓存 60s），防止端点注册后 DNS 重绑定绕过注册时校验
        if not await is_safe_url(url):
            raise A2AClientError("A2A 端点 URL 不安全（https + 非内网）")
        try:
            resp = await self._client.post(url, json=payload, headers=self._build_headers(endpoint))
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise A2AClientError(f"A2A 调用 {method} 失败: {e}") from e
        if body.get("error"):
            raise A2AClientError(f"A2A {method} 返回错误: {body['error']}")
        result = body.get("result")
        if result is None:
            raise A2AClientError(f"A2A {method} 无 result")
        return result

    async def message_send(
        self,
        endpoint: SysAiAgentEndpoint,
        messages: list[dict],
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> Task:
        """message/send：发送消息，返回 Task（异步，需轮询 tasks/get）。"""
        params = {"messages": messages}
        if task_id:
            params["taskId"] = task_id
        if context_id:
            params["contextId"] = context_id
        result = await self._rpc(endpoint, "message/send", params)
        return Task.model_validate(result)

    async def task_get(self, endpoint: SysAiAgentEndpoint, task_id: str) -> Task:
        result = await self._rpc(endpoint, "tasks/get", {"taskId": task_id})
        return Task.model_validate(result)

    async def task_cancel(self, endpoint: SysAiAgentEndpoint, task_id: str) -> dict:
        return await self._rpc(endpoint, "tasks/cancel", {"taskId": task_id})

    async def message_stream(
        self,
        endpoint: SysAiAgentEndpoint,
        messages: list[dict],
        task_id: str | None = None,
        context_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """message/stream：SSE 流式解析事件
        （status-update / artifact-update / message / error）。
        """
        payload = JsonRpcRequest(
            id=1,
            method="message/stream",
            params={
                "messages": messages,
                **({"taskId": task_id} if task_id else {}),
                **({"contextId": context_id} if context_id else {}),
            },
        ).model_dump(exclude_none=True)
        url = self._endpoint_url(endpoint)
        if not await is_safe_url(url):
            raise A2AClientError("A2A 端点 URL 不安全（https + 非内网）")
        try:
            async with self._client.stream(
                "POST", url, json=payload, headers=self._build_headers(endpoint)
            ) as resp:
                resp.raise_for_status()
                event = None
                data_parts: list[str] = []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        # 上一事件收尾
                        if event and data_parts:
                            yield {"event": event, "data": _parse_sse_data(data_parts)}
                        event = line[len("event:") :].strip()
                        data_parts = []
                    elif line.startswith("data:"):
                        data_parts.append(line[len("data:") :].strip())
                    elif line == "" and event:
                        # 空行结束当前事件
                        yield {"event": event, "data": _parse_sse_data(data_parts)}
                        event = None
                        data_parts = []
                if event and data_parts:
                    yield {"event": event, "data": _parse_sse_data(data_parts)}
        except httpx.HTTPError as e:
            raise A2AClientError(f"A2A message/stream 失败: {e}") from e


def _parse_sse_data(data_parts: list[str]) -> dict:
    """解析 SSE data 多行 JSON。"""
    data = "\n".join(data_parts)
    if not data:
        return {}
    try:
        parsed = json.loads(data)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        return {"raw": data}


a2a_client = A2AClient()
