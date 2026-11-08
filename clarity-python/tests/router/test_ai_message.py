"""AI 消息路由测试：SSE 流式发送 / 幂等键 / 续流 / 停止 / 重生成 / 编辑 / 分支 / 详情

覆盖重点：
- SSE 流式事件流（mock 服务层流式生成器，校验事件帧与响应头）
- Idempotency-Key 幂等（pending 命中 A0002、已完成命中复用原消息）
- 业务错误码透传（A0401/A0502）
- 消息详情可观测性扩展字段（traceId/contextSnapshot/llmCalls）
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.database import get_db
from app.dependencies.auth import get_current_user
from app.infrastructure.sse.sse_emitter_manager import sse_emitter_manager
from app.main import app as fastapi_app
from app.models.schema.ai_contract_gen import Message
from app.models.schema.ai_conversation import (
    ConversationResult,
    MessageListResult,
)
from app.service.ai_conversation_service import ai_conversation_service
from app.service.ai_message_service import ai_message_service

pytestmark = pytest.mark.api


_IDEMPOTENT_PREFIX = "ai:msg:idempotent:"


class _FakeUser:
    def __init__(self, id=1, is_root=False, permissions=()):
        self.id = id
        self.is_root = is_root
        self.permissions = list(permissions)


def _message(**overrides) -> Message:
    """构造契约 Message（内容在 blocks，用量在 usage）。"""
    base = {
        "id": 55,
        "conversationId": 3,
        "parentMessageId": None,
        "role": "assistant",
        "status": "completed",
        "model": "gpt-4o-mini",
        "error": None,
        "blocks": [
            {
                "id": "b1",
                "kind": "text",
                "seq": 1,
                "text": "你好",
                "durationMs": 12,
            }
        ],
        "plan": None,
        "pendingInterrupt": None,
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "cachedInputTokens": 0,
            "credits": 3,
        },
        "artifactIds": [],
        "memoryIds": [],
        "suggestions": [],
        "edited": False,
        "createTime": "2026-08-29T10:00:00",
        "updateTime": None,
    }
    base.update(overrides)
    return Message.model_validate(base)


def _msg_entity(**overrides) -> SimpleNamespace:
    """消息实体桩（读模型 build_message 消费 ORM 行形状）。"""
    base = {
        "id": 55,
        "conversation_id": 3,
        "parent_message_id": None,
        "role": "assistant",
        "status": "completed",
        "model": "gpt-4o-mini",
        "error": None,
        "blocks": [{"id": "b1", "kind": "text", "seq": 1, "text": "你好", "durationMs": 12}],
        "plan": None,
        "pending_interrupt": None,
        "suggestions": None,
        "used_memory_ids": None,
        "input_tokens": 10,
        "output_tokens": 20,
        "cached_input_tokens": 0,
        "credits": 3,
        "edited": 0,
        "create_time": datetime(2026, 8, 29, 10, 0, 0),
        "update_time": None,
    }
    base.update(overrides)
    entity = SimpleNamespace(**base)
    entity.text = "".join(
        b["text"]
        for b in sorted((b for b in entity.blocks if b["kind"] == "text"), key=lambda b: b["seq"])
    )
    return entity


def _no_artifacts(monkeypatch) -> None:
    """读模型会批量查询产物 ID：路由测试的 db 是桩，产物查询直接短路为空。"""

    async def _empty(db, msg_ids):
        return []

    monkeypatch.setattr(
        "app.service.ai.service.message_view.ai_artifact_repository.list_by_message_ids", _empty
    )


def _conversation(**overrides) -> ConversationResult:
    base = {
        "id": 3,
        "user_id": 8,
        "title": "测试会话",
        "message_count": 4,
        "pinned": 0,
        "title_source": "auto",
        "status": 1,
    }
    base.update(overrides)
    return ConversationResult.model_validate(base)


def _sse_stream(events: list[str]) -> StreamingResponse:
    async def _gen():
        for event in events:
            yield event

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@pytest.fixture
async def msg_client():
    async def _override_db():
        return object()

    current_user = {"user": _FakeUser()}

    async def _override_user():
        return current_user["user"]

    fastapi_app.dependency_overrides[get_db] = _override_db
    fastapi_app.dependency_overrides[get_current_user] = _override_user
    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client, current_user
    fastapi_app.dependency_overrides.pop(get_db, None)
    fastapi_app.dependency_overrides.pop(get_current_user, None)


def test_message_paths_registered(app):
    schema = app.openapi()
    for path in (
        "/api/v1/ai/conversations/{conv_id}/messages",
        "/api/v1/ai/conversations/{conv_id}/messages/stream/{stream_session_id}",
        "/api/v1/ai/conversations/{conv_id}/messages/{msg_id}/branches",
        "/api/v1/ai/conversations/{conv_id}/branches/{msg_id}",
        "/api/v1/ai/messages/{msg_id}",
        "/api/v1/ai/messages/{msg_id}/regenerate",
        "/api/v1/ai/messages/{msg_id}/resume",
        "/api/v1/ai/messages/{msg_id}/stop",
    ):
        assert path in schema["paths"], f"缺少路径 {path}"


class TestSendStream:
    async def test_send_streams_events_and_forwards_key(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_send(db, conv_id, user_id, form, idempotency_key):
            captured.update(
                conv_id=conv_id, user_id=user_id, content=form.content, key=idempotency_key
            )
            return _sse_stream(
                [
                    "event: message\ndata: " + json.dumps({"type": "start"}) + "\n\n",
                    "event: message\ndata: "
                    + json.dumps({"type": "text", "content": "你"})
                    + "\n\n",
                    "event: message\ndata: " + json.dumps({"type": "done"}) + "\n\n",
                ]
            )

        monkeypatch.setattr(ai_message_service, "send_message", _fake_send)
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-001"},
        )
        assert resp.status_code == 200
        assert captured == {
            "conv_id": 3,
            "user_id": 8,
            "content": "你好",
            "key": "key-001",
        }
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [
            block.split("data: ", 1)[1]
            for block in resp.text.split("\n\n")
            if block.strip().startswith("event:")
        ]
        assert [json.loads(e)["type"] for e in events] == ["start", "text", "done"]

    async def test_send_requires_idempotency_key(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_send(db, conv_id, user_id, form, idempotency_key):
            return _sse_stream([])

        monkeypatch.setattr(ai_message_service, "send_message", _fake_send)
        resp = await client.post("/api/v1/ai/conversations/3/messages", json={"content": "你好"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_send_empty_content_rejected(self, msg_client):
        client, _ = msg_client
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": ""},
            headers={"Idempotency-Key": "key-002"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_send_conversation_not_found_maps_a0401(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_send(db, conv_id, user_id, form, idempotency_key):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        monkeypatch.setattr(ai_message_service, "send_message", _fake_send)
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-003"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestIdempotency:
    """幂等键走真实 service 逻辑（Redis 用 fakeredis，仓储 mock）。"""

    async def test_pending_key_returns_a0002(self, msg_client, monkeypatch, mock_redis):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        await mock_redis.set(f"{_IDEMPOTENT_PREFIX}8:key-pending", "pending")

        async def _fake_conv(db, conv_id, user_id):
            return _FakeConv()

        monkeypatch.setattr(
            ai_message_service.ai_conversation_repository, "get_by_id_and_user", _fake_conv
        )
        monkeypatch.setattr(ai_message_service, "get_redis_client", _fake_redis(mock_redis))
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-pending"},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0002"

    async def test_completed_key_reuses_original_message(self, msg_client, monkeypatch, mock_redis):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        await mock_redis.set(f"{_IDEMPOTENT_PREFIX}8:key-done", json.dumps({"messageId": 55}))

        async def _fake_conv(db, conv_id, user_id):
            return _FakeConv()

        async def _fake_msg(db, msg_id):
            captured_ids.append(msg_id)
            return _msg_entity(id=msg_id)

        captured_ids: list[int] = []
        monkeypatch.setattr(
            ai_message_service.ai_conversation_repository, "get_by_id_and_user", _fake_conv
        )
        monkeypatch.setattr(ai_message_service.ai_message_repository, "get_by_id", _fake_msg)
        monkeypatch.setattr(ai_message_service, "get_redis_client", _fake_redis(mock_redis))
        _no_artifacts(monkeypatch)
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-done"},
        )
        assert resp.status_code == 200
        assert captured_ids == [55]
        assert resp.json()["data"]["id"] == 55
        assert resp.json()["data"]["role"] == "assistant"

    async def test_completed_key_replays_message_with_create_time(
        self, msg_client, monkeypatch, mock_redis
    ):
        """缓存消息的 create_time 为 datetime：重放须序列化为 ISO 串返回，不得吞掉 data"""
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        await mock_redis.set(f"{_IDEMPOTENT_PREFIX}8:key-done-time", json.dumps({"messageId": 56}))

        async def _fake_conv(db, conv_id, user_id):
            return _FakeConv()

        async def _fake_msg(db, msg_id):
            return _msg_entity(id=msg_id)

        monkeypatch.setattr(
            ai_message_service.ai_conversation_repository, "get_by_id_and_user", _fake_conv
        )
        monkeypatch.setattr(ai_message_service.ai_message_repository, "get_by_id", _fake_msg)
        monkeypatch.setattr(ai_message_service, "get_redis_client", _fake_redis(mock_redis))
        _no_artifacts(monkeypatch)
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-done-time"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["id"] == 56
        assert resp.json()["data"]["createTime"] == "2026-08-29T10:00:00"

    async def test_malformed_idempotent_cache_returns_empty_success(
        self, msg_client, monkeypatch, mock_redis
    ):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        await mock_redis.set(f"{_IDEMPOTENT_PREFIX}8:key-broken", "{not-json")

        async def _fake_conv(db, conv_id, user_id):
            return _FakeConv()

        monkeypatch.setattr(
            ai_message_service.ai_conversation_repository, "get_by_id_and_user", _fake_conv
        )
        monkeypatch.setattr(ai_message_service, "get_redis_client", _fake_redis(mock_redis))
        resp = await client.post(
            "/api/v1/ai/conversations/3/messages",
            json={"content": "你好"},
            headers={"Idempotency-Key": "key-broken"},
        )
        assert resp.status_code == 200
        assert "data" not in resp.json()


class _FakeConv:
    """会话桩：活跃会话、无中断挂起，供 send_message 走通幂等判定"""

    id = 3
    status = 1
    current_branch_message_id = None


def _fake_redis(redis):
    async def _provide():
        return redis

    return _provide


class TestResumeAndStop:
    async def test_resume_forwards_confirm(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_resume(db, msg_id, user_id, form):
            captured.update(msg_id=msg_id, user_id=user_id, confirm=form.confirm)
            return _sse_stream(['event: message\ndata: {"type": "done"}\n\n'])

        monkeypatch.setattr(ai_conversation_service, "resume_message", _fake_resume)
        resp = await client.post("/api/v1/ai/messages/55/resume", json={"confirm": True})
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8, "confirm": True}
        assert resp.headers["content-type"].startswith("text/event-stream")

    async def test_resume_without_interrupt_maps_a0401(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_resume(db, msg_id, user_id, form):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "未找到中断点，无法恢复")

        monkeypatch.setattr(ai_conversation_service, "resume_message", _fake_resume)
        resp = await client.post("/api/v1/ai/messages/55/resume", json={"confirm": True})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_stop_returns_message(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_stop(db, msg_id, user_id):
            captured.update(msg_id=msg_id, user_id=user_id)
            return _message(id=msg_id, status="canceled")

        monkeypatch.setattr(ai_conversation_service, "stop_message", _fake_stop)
        resp = await client.post("/api/v1/ai/messages/55/stop")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8}
        assert resp.json()["data"]["status"] == "canceled"

    async def test_stop_not_streaming_maps_a0502(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_stop(db, msg_id, user_id):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "当前消息不可停止")

        monkeypatch.setattr(ai_conversation_service, "stop_message", _fake_stop)
        resp = await client.post("/api/v1/ai/messages/55/stop")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"


class TestRegenerateAndEdit:
    async def test_regenerate_streams(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_regen(db, msg_id, user_id):
            captured.update(msg_id=msg_id, user_id=user_id)
            return _sse_stream(['event: message\ndata: {"type": "text", "content": "hi"}\n\n'])

        monkeypatch.setattr(ai_conversation_service, "regenerate_message", _fake_regen)
        resp = await client.post("/api/v1/ai/messages/55/regenerate")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8}
        assert "hi" in resp.text

    async def test_regenerate_user_message_maps_a0502(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_regen(db, msg_id, user_id):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可重新生成")

        monkeypatch.setattr(ai_conversation_service, "regenerate_message", _fake_regen)
        resp = await client.post("/api/v1/ai/messages/55/regenerate")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"

    async def test_edit_streams_and_forwards_content(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_edit(db, user_id, msg_id, form):
            captured.update(user_id=user_id, msg_id=msg_id, content=form.content)
            return _sse_stream(['event: message\ndata: {"type": "done"}\n\n'])

        monkeypatch.setattr(ai_message_service, "edit_message", _fake_edit)
        resp = await client.put("/api/v1/ai/messages/54", json={"content": "改后的提问"})
        assert resp.status_code == 200
        assert captured == {"user_id": 8, "msg_id": 54, "content": "改后的提问"}

    async def test_edit_empty_content_rejected(self, msg_client):
        client, _ = msg_client
        resp = await client.put("/api/v1/ai/messages/54", json={"content": ""})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_edit_over_model_budget_maps_a0500(self, msg_client, monkeypatch):
        """单条消息长度上限按模型上下文预算在服务层判定（A0500），路由原样透传

        长度上限不再是固定的 4000 字符 schema 校验，故此处只断言错误码映射。
        """
        client, _ = msg_client

        async def _fake_edit(db, user_id, msg_id, form):
            raise BusinessException(
                ResultCode.BUSINESS_ERROR, "单条消息超出模型上下文预算（800 字符，模型 gpt-4o）"
            )

        monkeypatch.setattr(ai_message_service, "edit_message", _fake_edit)
        resp = await client.put("/api/v1/ai/messages/54", json={"content": "x" * 801})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0500"

    async def test_edit_assistant_message_maps_a0502(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_edit(db, user_id, msg_id, form):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅用户消息可编辑")

        monkeypatch.setattr(ai_message_service, "edit_message", _fake_edit)
        resp = await client.put("/api/v1/ai/messages/55", json={"content": "改后"})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"


class TestBranches:
    async def test_list_branches(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_branches(db, conv_id, user_id, msg_id):
            captured.update(conv_id=conv_id, user_id=user_id, msg_id=msg_id)
            return [_message(id=msg_id), _message(id=99)]

        monkeypatch.setattr(ai_conversation_service, "get_branches", _fake_branches)
        resp = await client.get("/api/v1/ai/conversations/3/messages/55/branches")
        assert resp.status_code == 200
        assert captured == {"conv_id": 3, "user_id": 8, "msg_id": 55}
        assert [item["id"] for item in resp.json()["data"]] == [55, 99]

    async def test_list_branches_message_not_found_maps_a0401(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_branches(db, conv_id, user_id, msg_id):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")

        monkeypatch.setattr(ai_conversation_service, "get_branches", _fake_branches)
        resp = await client.get("/api/v1/ai/conversations/3/messages/55/branches")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"

    async def test_switch_branch(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_switch(db, conv_id, user_id, msg_id):
            captured.update(conv_id=conv_id, user_id=user_id, msg_id=msg_id)
            return _conversation(id=conv_id, current_branch_message_id=msg_id)

        monkeypatch.setattr(ai_conversation_service, "switch_branch", _fake_switch)
        resp = await client.put("/api/v1/ai/conversations/3/branches/99")
        assert resp.status_code == 200
        assert captured == {"conv_id": 3, "user_id": 8, "msg_id": 99}
        assert resp.json()["data"]["currentBranchMessageId"] == 99


class TestMessageDetail:
    async def test_detail_returns_contract_message(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_detail(db, msg_id, user_id, admin=False):
            captured.update(msg_id=msg_id, user_id=user_id, admin=admin)
            return _message(id=msg_id).model_dump(by_alias=True, mode="json")

        monkeypatch.setattr(ai_conversation_service, "get_message", _fake_detail)
        resp = await client.get("/api/v1/ai/messages/55")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8, "admin": False}
        data = resp.json()["data"]
        assert data["blocks"][0]["kind"] == "text"
        assert data["blocks"][0]["text"] == "你好"
        assert data["usage"]["outputTokens"] == 20
        # 旧字段与旧投影一律不再出现在出参中
        for legacy in ("content", "thoughts", "toolCalls", "tool_calls", "metadata"):
            assert legacy not in data
        # 消息详情不再附带过程链扩展（可观测下钻走 /traces/{traceId} 与会话时间线）
        assert "traceId" not in data
        assert "contextSnapshot" not in data
        assert "llmCalls" not in data

    async def test_detail_without_blocks(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_detail(db, msg_id, user_id, admin=False):
            return _message(id=msg_id, blocks=[]).model_dump(by_alias=True, mode="json")

        monkeypatch.setattr(ai_conversation_service, "get_message", _fake_detail)
        resp = await client.get("/api/v1/ai/messages/55")
        assert resp.status_code == 200
        assert resp.json()["data"]["blocks"] == []

    async def test_interrupt_endpoint_returns_null_without_pending(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_interrupt(db, msg_id, user_id, admin=False):
            captured.update(msg_id=msg_id, user_id=user_id, admin=admin)
            return

        monkeypatch.setattr(ai_conversation_service, "get_interrupt", _fake_interrupt)
        resp = await client.get("/api/v1/ai/messages/55/interrupt")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8, "admin": False}
        # 全局 NON_NULL 序列化（与 Java Jackson NON_NULL 对齐）：无中断时 data 缺省即 null
        assert resp.json().get("data") is None

    async def test_interrupt_endpoint_returns_pending(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)

        async def _fake_interrupt(db, msg_id, user_id, admin=False):
            return _message(
                id=msg_id,
                status="awaiting_input",
                pendingInterrupt={
                    "id": "i1",
                    "type": "confirm",
                    "confirmKind": "algorithm_recommend",
                    "createdAt": "2026-08-29T10:00:00",
                    "expiresAt": "2026-09-28T10:00:00",
                    "actionable": False,
                },
            ).pendingInterrupt

        monkeypatch.setattr(ai_conversation_service, "get_interrupt", _fake_interrupt)
        resp = await client.get("/api/v1/ai/messages/55/interrupt")
        assert resp.status_code == 200
        assert resp.json()["data"]["type"] == "confirm"
        assert resp.json()["data"]["actionable"] is False

    async def test_detail_not_found_maps_a0401(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_detail(db, msg_id, user_id, admin=False):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在")

        monkeypatch.setattr(ai_conversation_service, "get_message", _fake_detail)
        resp = await client.get("/api/v1/ai/messages/55")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestDeleteMessage:
    async def test_delete_assistant_message(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_delete(db, msg_id, user_id):
            captured.update(msg_id=msg_id, user_id=user_id)

        monkeypatch.setattr(ai_conversation_service, "delete_message", _fake_delete)
        resp = await client.delete("/api/v1/ai/messages/55")
        assert resp.status_code == 200
        assert captured == {"msg_id": 55, "user_id": 8}

    async def test_delete_user_message_maps_a0502(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_delete(db, msg_id, user_id):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可删除")

        monkeypatch.setattr(ai_conversation_service, "delete_message", _fake_delete)
        resp = await client.delete("/api/v1/ai/messages/54")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"

    async def test_delete_non_leaf_maps_a0502_with_guidance(self, msg_client, monkeypatch):
        """中间回复仍有后续对话：拒绝删除并给出明确指引，客户端按 400 处理。"""
        client, _ = msg_client

        async def _fake_delete(db, msg_id, user_id):
            raise BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "请先删除该回复之后的对话")

        monkeypatch.setattr(ai_conversation_service, "delete_message", _fake_delete)
        resp = await client.delete("/api/v1/ai/messages/55")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0502"
        assert "请先删除该回复之后的对话" in resp.json()["msg"]


class TestReconnect:
    async def test_reconnect_checks_ownership_and_forwards_last_event_id(
        self, msg_client, monkeypatch
    ):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_get(db, conv_id, user_id, admin=False):
            captured.update(conv_id=conv_id, user_id=user_id)
            return _conversation(id=conv_id)

        def _fake_reconnect(stream_session_id, last_event_id):
            captured.update(stream_session_id=stream_session_id, last_event_id=last_event_id)

            async def _gen():
                yield 'event: message\ndata: {"type": "done"}\n\n'

            return _gen()

        monkeypatch.setattr(ai_conversation_service, "get_conversation", _fake_get)
        monkeypatch.setattr(sse_emitter_manager, "reconnect", _fake_reconnect)
        resp = await client.get(
            "/api/v1/ai/conversations/3/messages/stream/sess-1",
            headers={"Last-Event-ID": "12"},
        )
        assert resp.status_code == 200
        assert captured == {
            "conv_id": 3,
            "user_id": 8,
            "stream_session_id": "sess-1",
            "last_event_id": 12,
        }
        assert resp.headers["content-type"].startswith("text/event-stream")

    async def test_reconnect_cross_user_conversation_maps_a0401(self, msg_client, monkeypatch):
        client, _ = msg_client

        async def _fake_get(db, conv_id, user_id, admin=False):
            raise BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在")

        monkeypatch.setattr(ai_conversation_service, "get_conversation", _fake_get)
        resp = await client.get("/api/v1/ai/conversations/3/messages/stream/sess-1")
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0401"


class TestMessageList:
    async def test_list_messages_forwards_cursor(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_list(db, conv_id, user_id, before, limit, admin=False):
            captured.update(
                conv_id=conv_id, user_id=user_id, before=before, limit=limit, admin=admin
            )
            return MessageListResult(list=[_message()], total=1, hasMore=True)

        monkeypatch.setattr(ai_conversation_service, "list_messages", _fake_list)
        resp = await client.get(
            "/api/v1/ai/conversations/3/messages", params={"before": 42, "limit": 5}
        )
        assert resp.status_code == 200
        assert captured == {"conv_id": 3, "user_id": 8, "before": 42, "limit": 5, "admin": False}
        body = resp.json()["data"]
        assert body["list"][0]["id"] == 55
        assert body["total"] == 1
        assert body["hasMore"] is True

    async def test_list_messages_default_limit(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        captured: dict = {}

        async def _fake_list(db, conv_id, user_id, before, limit, admin=False):
            captured.update(before=before, limit=limit)
            return MessageListResult(list=[], total=0, hasMore=False)

        monkeypatch.setattr(ai_conversation_service, "list_messages", _fake_list)
        resp = await client.get("/api/v1/ai/conversations/3/messages")
        assert resp.status_code == 200
        assert captured == {"before": None, "limit": 50}

    @pytest.mark.parametrize("limit", [0, 101])
    async def test_list_messages_limit_out_of_range_maps_a0400(self, msg_client, limit):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)
        resp = await client.get("/api/v1/ai/conversations/3/messages", params={"limit": limit})
        assert resp.status_code == 400
        assert resp.json()["code"] == "A0400"

    async def test_list_messages_view_admin_forwards_admin_flag(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8, permissions=["ai:conversation:audit"])
        captured: dict = {}

        async def _fake_list(db, conv_id, user_id, before, limit, admin=False):
            captured.update(
                conv_id=conv_id, user_id=user_id, before=before, limit=limit, admin=admin
            )
            return MessageListResult(list=[_message()], total=1, hasMore=False)

        monkeypatch.setattr(ai_conversation_service, "list_messages", _fake_list)
        resp = await client.get(
            "/api/v1/ai/conversations/3/messages",
            params={"before": 10, "limit": 10, "view": "admin"},
        )
        assert resp.status_code == 200
        assert captured == {"conv_id": 3, "user_id": 8, "before": 10, "limit": 10, "admin": True}

    async def test_list_messages_view_admin_requires_permission(self, msg_client, monkeypatch):
        client, state = msg_client
        state["user"] = _FakeUser(id=8)  # 无 ai:conversation:audit

        resp = await client.get(
            "/api/v1/ai/conversations/3/messages",
            params={"view": "admin"},
        )
        assert resp.status_code == 403
        assert resp.json()["code"] == "A0301"
