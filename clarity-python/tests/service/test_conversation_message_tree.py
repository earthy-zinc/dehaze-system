from datetime import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.code import ResultCode
from app.core.exceptions import BusinessException
from app.infrastructure.sse.sse_emitter_manager import SseEmitterManager
from app.models.schema.ai_conversation import MessageResume
from app.repository.ai_conversation_repository import (
    AiConversationRepository,
    ai_conversation_repository,
)
from app.repository.ai_message_repository import ai_message_repository as ai_msg_repo
from app.service.ai import message_streaming
from app.service.ai.builders.context_manager import context_manager
from app.service.ai.middleware.interrupt_handler import InterruptHandler
from app.service.ai.service.reasoning_service import ReasoningService
from app.service.ai_conversation_service import AiConversationService, ai_conversation_service
from app.service.ai_message_service import AiMessageService, ai_message_service
from tests.stubs.factories import make_conv
from tests.stubs.mocks import patch_reasoning_boundaries

pytestmark = pytest.mark.requires_db


def _patch_artifacts(monkeypatch) -> None:
    """读模型会批量查产物 ID：本文件的 db 为未绑定 session，产物查询短路为空。"""

    async def _empty(db, msg_ids):
        return []

    monkeypatch.setattr(
        "app.service.ai.service.message_view.ai_artifact_repository.list_by_message_ids", _empty
    )


def _msg(id, conv_id, parent, role, content="", **kw):
    """消息桩：正文在 blocks 的 text 块里，状态为契约字符串枚举。"""
    base = {
        "id": id,
        "conversation_id": conv_id,
        "parent_message_id": parent,
        "role": role,
        "deleted": 0,
        "blocks": [{"id": f"b{id}", "kind": "text", "seq": 1, "text": content}],
        "model": "gpt",
        "status": "completed",
        "task_id": None,
        "error": None,
        "plan": None,
        "pending_interrupt": None,
        "suggestions": None,
        "used_memory_ids": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "credits": 0,
        "edited": 0,
        "create_time": None,
        "update_time": None,
    }
    base.update(kw)
    entity = SimpleNamespace(**base)
    entity.text = content
    return entity


class _MsgRepo:
    """内存版 ai_message_repository 桩：按 id 存 SimpleNamespace 消息"""

    def __init__(self, next_id=100):
        self.msgs = {}
        self.created = []
        self._next_id = next_id

    def put(self, msg):
        self.msgs[msg.id] = msg
        return self

    async def get_by_id_and_user(self, db, mid, uid):
        return self.msgs.get(mid)

    async def get_by_id(self, db, mid, *, with_deleted=False):
        msg = self.msgs.get(mid)
        if msg is None or (not with_deleted and getattr(msg, "deleted", 0)):
            return None
        return msg

    async def create(self, db, entity):
        entity.id = self._next_id
        self._next_id += 1
        self.msgs[entity.id] = entity
        self.created.append(entity)
        return entity

    async def get_children(self, db, cid, pid, role=None):
        return [
            m
            for m in self.msgs.values()
            if m.conversation_id == cid
            and m.parent_message_id == pid
            and not getattr(m, "deleted", 0)
            and (role is None or m.role == role)
        ]

    async def list_assistant_sibling_ids(self, db, cid, parent_ids):
        grouped = {}
        for m in sorted(self.msgs.values(), key=lambda m: m.id):
            if (
                m.conversation_id == cid
                and m.parent_message_id in parent_ids
                and m.role == "assistant"
                and not getattr(m, "deleted", 0)
            ):
                grouped.setdefault(m.parent_message_id, []).append(m.id)
        return grouped

    async def soft_delete_by_ids(self, db, ids):
        for i in ids:
            self.msgs[i].deleted = 1

    async def update_status(self, db, mid, status):
        self.msgs[mid].status = status

    async def update_task_id(self, db, mid, task_id):
        self.msgs[mid].task_id = task_id

    async def update_pending_interrupt(self, db, mid, pending):
        self.msgs[mid].pending_interrupt = pending


class _ConvRepo:
    def __init__(self, conv=None):
        self._conv = conv

    async def get_by_id_and_user(self, db, cid, uid, *, for_update=False):
        return self._conv

    async def update_last_message(self, db, cid, mid, t, **kw):
        if self._conv is not None:
            self._conv.current_branch_message_id = mid

    async def update_current_branch(self, db, cid, mid):
        if self._conv is not None:
            self._conv.current_branch_message_id = mid


class _ReasoningStub(ReasoningService):
    """测试替身：仅实现 run（委托注入的协程），其余推理能力不参与本组用例。"""

    def __init__(self, run):
        self._run = run

    async def run(self, conv_id, user_id, msg_id, model_id, stream_session_id):
        await self._run(
            conv_id=conv_id,
            user_id=user_id,
            msg_id=msg_id,
            model_id=model_id,
            stream_session_id=stream_session_id,
        )
        return {}


class _EmitterStub(SseEmitterManager):
    """测试替身：仅实现 stop_stream / acquire_lock 中被注入者，其余推流能力不参与。"""

    def __init__(self, *, stop_stream=None, acquire_lock=None):
        self._stop_stream = stop_stream
        self._acquire_lock = acquire_lock

    async def stop_stream(self, stream_session_id):
        if self._stop_stream is not None:
            await self._stop_stream(stream_session_id)

    async def acquire_lock(self, conversation_id):
        if self._acquire_lock is not None:
            return await self._acquire_lock(conversation_id)
        return True


class _InterruptStub(InterruptHandler):
    """测试替身：仅实现 get_interrupt，返回注入的中断点（None 表示无中断）。"""

    def __init__(self, result):
        self._result = result

    async def get_interrupt(self, thread_id):
        return self._result


def _conv_service(**kw):
    """构造 AiConversationService：未显式注入的依赖回落模块级单例（测试中仅用桩字段）"""
    return AiConversationService(**kw)


def _switch_conv(**overrides):
    """switch_branch 成功路径序列化 ConversationResult，桩会话需补齐契约字段。"""
    fields = {
        "title": "会话",
        "model": "gpt",
        "message_count": 5,
        "model_config": None,
        "api_key_id": None,
        "last_message_at": None,
        "last_read_message_id": None,
        "pinned": 0,
        "pinned_at": None,
        "delete_time": None,
        "unread_count": 0,
        "title_source": "auto",
        "create_time": None,
        "update_time": None,
    }
    fields.update(overrides)
    return make_conv(**fields)


class TestRegenerate:
    async def test_branch_structure_and_wiring(self, monkeypatch):
        user_msg = _msg(1, 10, None, "user", content="你好")
        old_asst = _msg(2, 10, 1, "assistant", content="旧回复")
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        wired = {}

        async def fake_stream(**kw):
            wired["assistant_msg_id"] = kw["assistant_msg_id"]
            wired["stream_session_id"] = kw["stream_session_id"]
            yield b"data"

        svc = _conv_service(
            ai_message_repository=_MsgRepo(next_id=3).put(user_msg).put(old_asst),
            ai_conversation_repository=_ConvRepo(conv),
            sse_emitter_manager=SimpleNamespace(acquire_lock=AsyncMock(return_value=True)),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
        )
        monkeypatch.setattr(message_streaming, "stream_generator", fake_stream)

        resp = await svc.regenerate_message(AsyncSession(), 2, 1)
        # 断言面取自注入的 _MsgRepo 替身（svc.ai_message_repository 即该替身，验证装配生效）
        repo = svc.ai_message_repository
        assert isinstance(repo, _MsgRepo)
        created = repo.created
        assert created[0].role == "assistant"
        assert created[0].parent_message_id == 1
        assert created[0].id == 3
        assert conv.current_branch_message_id == 3
        assert resp.media_type == "text/event-stream"
        async for _ in resp.body_iterator:
            pass
        assert wired["assistant_msg_id"] == 3

    async def test_rejects_deleted_message(self):
        deleted_asst = _msg(2, 10, 1, "assistant", deleted=1)
        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(deleted_asst),
        )
        with pytest.raises(BusinessException):
            await svc.regenerate_message(AsyncSession(), 2, 1)

    async def test_shared_reasoning_trigger(self, mock_redis):
        run_calls = {}

        async def fake_run(**kw):
            run_calls.update(kw)

        async def fake_stop(sid):
            pass

        svc = AiMessageService(
            # 测试替身：仅实现被测方法（reasoning_service.run / emitter.stop_stream）
            reasoning_service=_ReasoningStub(fake_run),
            sse_emitter_manager=_EmitterStub(stop_stream=fake_stop),
            get_redis_client=lambda: mock_redis,
        )

        await svc._run_reasoning(
            conv_id=10,
            user_id=1,
            model="gpt",
            assistant_msg_id=3,
            stream_session_id="s1",
            idem_key="k",
        )
        assert run_calls["conv_id"] == 10
        assert run_calls["msg_id"] == 3
        assert run_calls["model_id"] == "gpt"
        assert await mock_redis.get("k")
        assert run_calls["stream_session_id"] == "s1"


class TestBranchChain:
    @staticmethod
    async def _seed(db, specs):
        """落库真实会话与消息：specs 为 [(role, parent_index|None, content), ...]"""
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt", agent_code=None)
        db.add(conv)
        await db.flush()
        msgs = []
        for role, parent_index, content in specs:
            msg = SysAiMessage(
                conversation_id=conv.id,
                parent_message_id=msgs[parent_index].id if parent_index is not None else None,
                role=role,
                status="completed",
                blocks=[{"id": f"b{len(msgs)}", "kind": "text", "seq": 1, "text": content}],
            )
            db.add(msg)
            await db.flush()
            msgs.append(msg)
        return conv, msgs

    async def test_no_cross_branch_contamination(self, db):
        conv, msgs = await self._seed(
            db,
            [
                ("user", None, "a"),
                ("assistant", 0, "branchA"),
                ("user", 0, "b"),
                ("assistant", 2, "branchB"),
            ],
        )
        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[-1].id)
        assert [m.text for m in chain] == ["a", "b", "branchB"]

    async def test_ring_protection(self, db):
        conv, msgs = await self._seed(db, [("user", None, "x"), ("assistant", None, "y")])
        msgs[0].parent_message_id = msgs[1].id
        msgs[1].parent_message_id = msgs[0].id
        await db.flush()

        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[0].id)
        assert [m.text for m in chain] == ["y", "x"]

    async def test_limit_respected(self, db):
        conv, msgs = await self._seed(
            db, [("user", i - 1 if i else None, str(i + 1)) for i in range(49)]
        )
        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[-1].id, limit=20)
        assert len(chain) == 20
        assert chain[0].text == "30"
        assert chain[-1].text == "49"

    async def test_deleted_ancestor_terminates_chain(self, db):
        conv, msgs = await self._seed(
            db,
            [
                ("user", None, "a"),
                ("assistant", 0, "b"),
                ("user", 1, "c"),
                ("assistant", 2, "d"),
            ],
        )
        await ai_msg_repo.soft_delete_by_ids(db, [msgs[2].id])

        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[3].id)
        assert [m.text for m in chain] == ["d"]

    async def test_empty_and_single_message_chain(self, db):
        conv, msgs = await self._seed(db, [("user", None, "only")])
        assert await ai_msg_repo.get_chain_by_id(db, conv.id, None) == []
        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[0].id)
        assert [m.text for m in chain] == ["only"]

    async def test_full_chain_not_silently_truncated_over_1000_hops(self, db):
        """列表/导出全量回溯不设 hop 上限：超过旧 1000 兜底也须返回完整历史。"""
        from sqlalchemy import func, select

        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt")
        db.add(conv)
        await db.flush()
        count = 1002
        # 主键接在库内最大值之后：一条 add_all 落库避免逐条 flush，也不推高自增段
        max_id = (
            await db.execute(
                select(func.max(SysAiMessage.id)).execution_options(include_deleted=True)
            )
        ).scalar() or 0
        base = max_id + 1
        db.add_all(
            [
                SysAiMessage(
                    id=base + i,
                    conversation_id=conv.id,
                    parent_message_id=base + i - 1 if i else None,
                    role="user" if i % 2 == 0 else "assistant",
                    status="completed",
                    blocks=[],
                )
                for i in range(count)
            ]
        )
        await db.flush()

        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, base + count - 1, limit=None)
        assert len(chain) == count
        assert chain[0].id == base
        assert chain[-1].id == base + count - 1

    async def test_chain_uses_single_query(self, db):
        """长链只发一条 SQL：逐条回溯会退化成与链长等量的往返，是推理热路径的 N+1"""
        from sqlalchemy import event as sa_event

        conv, msgs = await self._seed(
            db, [("user", i - 1 if i else None, str(i + 1)) for i in range(49)]
        )
        statements = []

        def _record(state):
            statements.append(state.statement)

        sa_event.listen(db.sync_session, "do_orm_execute", _record)
        try:
            chain = await ai_msg_repo.get_chain_by_id(db, conv.id, msgs[-1].id, limit=None)
        finally:
            sa_event.remove(db.sync_session, "do_orm_execute", _record)

        assert len(chain) == 49
        assert len(statements) == 1

    async def test_context_load_uses_branch_chain(self, db, monkeypatch):
        conv, msgs = await self._seed(
            db,
            [
                ("user", None, "a"),
                ("assistant", 0, "branchA"),
                ("user", 0, "b"),
                ("assistant", 2, "branchB"),
            ],
        )
        conv.current_branch_message_id = msgs[-1].id
        await db.flush()

        monkeypatch.setattr(
            "app.service.ai.builders.context_manager.inject_memories",
            AsyncMock(return_value=(None, [])),
        )

        # 模型不在注册表 → 预算走保守兜底窗口（32768），短链不会被裁剪
        async def get_model(db, model_id):
            return None

        from app.repository.ai_model_repository import ai_model_repository

        monkeypatch.setattr(ai_model_repository, "get_by_model_id", get_model)
        messages, _system_prompt, _injected = await context_manager.build_context(db, conv, "gpt")
        assert [{k: v for k, v in m.items() if k != "id"} for m in messages] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "branchB"},
        ]
        assert all(m["content"] != "branchA" for m in messages)


PENDING = {
    "id": "i1",
    "type": "confirm",
    "confirmKind": "algorithm_recommend",
    "createdAt": "2026-01-01T00:00:00",
    "expiresAt": "2099-01-01T00:00:00",
    "actionable": True,
}


class TestResumeStop:
    async def test_resume_triggers_reasoning(self, monkeypatch, mock_redis):
        _patch_artifacts(monkeypatch)
        msg = _msg(5, 10, 3, "assistant", status="awaiting_input", pending_interrupt=PENDING)
        resume_calls = []

        async def fake_resume(conv_id, user_id, msg_id, resume_data, stream_session_id):
            resume_calls.append((conv_id, user_id, msg_id, resume_data, stream_session_id))

        async def fake_create_stream(cid, sid):
            yield b"data: 1"

        async def fake_stop_stream(sid):
            return None

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            sse_emitter_manager=SimpleNamespace(
                acquire_lock=AsyncMock(return_value=True),
                register_stream=AsyncMock(return_value=None),
                create_stream=fake_create_stream,
                stop_stream=fake_stop_stream,
            ),
            reasoning_service=SimpleNamespace(resume=fake_resume),
            get_redis_client=lambda: mock_redis,
        )

        form = MessageResume(confirm=True, params={"modelId": 7})
        resp = await svc.resume_message(AsyncSession(), 5, 1, form)
        async for _ in resp.body_iterator:
            pass
        assert resp.media_type == "text/event-stream"
        assert resume_calls
        assert resume_calls[0][0:3] == (10, 1, 5)
        assert resume_calls[0][3] == {"confirmed": True, "modelId": 7}
        # 续流使用新的流会话（旧流已终结），由 send 时生成并透传给推理层
        assert resume_calls[0][4]
        # 新流会话已重绑到消息上，停止端点才能作用于本次续流
        bound = await mock_redis.get("ai:msg:stream:5")
        assert bound == resume_calls[0][4]
        # async_wait 遗留的批量任务 ID 已清空，停止不再指向已结束的后台任务
        assert msg.task_id is None

    async def test_resume_expired_interrupt_rejected(self, monkeypatch):
        _patch_artifacts(monkeypatch)
        expired = {**PENDING, "expiresAt": "2020-01-01T00:00:00"}
        msg = _msg(5, 10, 3, "assistant", status="awaiting_input", pending_interrupt=expired)
        svc = _conv_service(ai_message_repository=_MsgRepo().put(msg))
        with pytest.raises(BusinessException) as exc:
            await svc.resume_message(AsyncSession(), 5, 1, MessageResume(confirm=True))
        assert exc.value.code == ResultCode.DATA_STATE_NOT_ALLOW

    async def test_resume_no_interrupt(self, monkeypatch):
        _patch_artifacts(monkeypatch)
        msg = _msg(5, 10, 3, "assistant")
        svc = _conv_service(ai_message_repository=_MsgRepo().put(msg))
        with pytest.raises(BusinessException):
            await svc.resume_message(AsyncSession(), 5, 1, MessageResume(confirm=False))

    async def test_stop_calls_reasoning_stop(self, monkeypatch):
        _patch_artifacts(monkeypatch)
        msg = _msg(5, 10, 3, "assistant", status="running", task_id="s1")
        stop_calls = []

        async def fake_stop(conv_id, msg_id, sid):
            stop_calls.append((conv_id, msg_id, sid))

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            reasoning_service=SimpleNamespace(stop=fake_stop),
        )
        result = await svc.stop_message(AsyncSession(), 5, 1)
        assert stop_calls == [(10, 5, "s1")]
        assert result.status == "canceled"
        assert result.pendingInterrupt is None

    async def test_stop_returns_flushed_blocks(self, monkeypatch):
        """停止响应必须带推理侧已生成的块快照，不能拿库里的空 blocks 覆盖前端。"""
        _patch_artifacts(monkeypatch)
        msg = _msg(5, 10, 3, "assistant", content="", status="running", task_id="s1")
        snapshot = [
            {"id": "b9", "kind": "text", "seq": 1, "text": "已生成的一半", "durationMs": 12}
        ]

        async def fake_stop(conv_id, msg_id, sid):
            return snapshot

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            reasoning_service=SimpleNamespace(stop=fake_stop),
        )
        result = await svc.stop_message(AsyncSession(), 5, 1)

        assert result.status == "canceled"
        assert [b.text for b in result.blocks] == ["已生成的一半"]

    async def test_stop_awaiting_input_clears_pending(self, monkeypatch, mock_redis):
        """中断挂起态也允许停止：清 Redis 中断点与 DB pending，消息置取消态。"""
        _patch_artifacts(monkeypatch)
        msg = _msg(
            5,
            10,
            3,
            "assistant",
            content="已生成的内容",
            status="awaiting_input",
            pending_interrupt=PENDING,
        )
        cleared = []

        async def _clear(thread_id):
            cleared.append(thread_id)

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            interrupt_handler=SimpleNamespace(clear_interrupt=_clear),
            get_redis_client=lambda: mock_redis,
        )
        result = await svc.stop_message(AsyncSession(), 5, 1)

        assert result.status == "canceled"
        assert result.pendingInterrupt is None
        assert cleared == ["10:5"]
        assert [b.text for b in result.blocks] == ["已生成的内容"]


class TestSiblingFields:
    """停止/详情返回的实体与列表同口径：assistant 带兄弟序号，root 级与非 assistant 为 null。"""

    @staticmethod
    async def _seed_branches(db):
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt")
        db.add(conv)
        await db.flush()
        user = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=None,
            role="user",
            status="completed",
            blocks=[{"id": "u", "kind": "text", "seq": 1, "text": "问题"}],
        )
        db.add(user)
        await db.flush()
        first = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=user.id,
            role="assistant",
            status="completed",
            blocks=[{"id": "a1", "kind": "text", "seq": 1, "text": "旧回复"}],
        )
        second = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=user.id,
            role="assistant",
            status="running",
            task_id="s1",
            blocks=[{"id": "a2", "kind": "text", "seq": 1, "text": "新回复"}],
        )
        db.add_all([first, second])
        await db.flush()
        conv.current_branch_message_id = second.id
        await db.flush()
        return conv, user, first, second

    async def test_stop_returns_sibling_position(self, db, monkeypatch):
        _patch_artifacts(monkeypatch)
        _, _, _, second = await self._seed_branches(db)

        async def fake_stop(_conv_id, _msg_id, _sid):
            return None

        svc = _conv_service(reasoning_service=SimpleNamespace(stop=fake_stop))
        result = await svc.stop_message(db, second.id, 1)

        assert result.status == "canceled"
        assert (result.siblingIndex, result.siblingCount) == (2, 2)

    async def test_detail_includes_sibling_position(self, db, monkeypatch):
        _patch_artifacts(monkeypatch)
        _, user, _, second = await self._seed_branches(db)

        detail = await ai_conversation_service.get_message(db, second.id, 1)
        assert (detail["siblingIndex"], detail["siblingCount"]) == (2, 2)

        # 非 assistant（user）不参与兄弟计数
        user_detail = await ai_conversation_service.get_message(db, user.id, 1)
        assert user_detail["siblingIndex"] is None
        assert user_detail["siblingCount"] is None


class TestEditMessage:
    async def test_edit_updates_original_message_in_place(self, monkeypatch):
        """编辑原地更新原 user 消息：列表不出现第二条 user 气泡，回复挂原消息下。"""
        user_msg = _msg(1, 10, None, "user", content="旧问题")
        conv = make_conv(id=10, user_id=1, status=1, model="gpt")
        repo = _MsgRepo(next_id=3).put(user_msg)
        wired = {}

        async def fake_stream(**kw):
            wired.update(kw)
            yield b"data"

        async def _fits(_db, _content, _model):
            return None

        monkeypatch.setattr("app.service.ai_message_service.assert_message_fits_context", _fits)
        monkeypatch.setattr("app.service.ai_message_service.stream_generator", fake_stream)

        svc = AiMessageService(
            ai_message_repository=repo,
            ai_conversation_repository=_ConvRepo(conv),
            sse_emitter_manager=SimpleNamespace(acquire_lock=AsyncMock(return_value=True)),
        )
        resp = await svc.edit_message(AsyncSession(), 1, 1, SimpleNamespace(content="新问题"))
        async for _ in resp.body_iterator:
            pass

        assert user_msg.edited == 1
        assert [b["text"] for b in user_msg.blocks] == ["新问题"]
        # 只新建 assistant：不再复制一条 user 消息
        assert [m.role for m in repo.created] == ["assistant"]
        assert repo.created[0].parent_message_id == 1
        assert wired["user_msg_id"] == 1


class TestBranches:
    async def test_get_branches_lists_children(self, monkeypatch):
        _patch_artifacts(monkeypatch)
        children = [
            _msg(2, 10, 1, "assistant", content="主分支"),
            _msg(3, 10, 1, "assistant", content="regenerate分支"),
        ]
        msg = _msg(1, 10, None, "user", content="原始问题")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(make_conv(id=10, user_id=1)),
            ai_message_repository=_MsgRepo().put(msg).put(children[0]).put(children[1]),
        )

        result = await svc.get_branches(AsyncSession(), 10, 1, 1)
        assert [m.id for m in result] == [2, 3]

    async def test_get_branches_only_returns_assistant_children(self, monkeypatch):
        """分支切换器只列 assistant 回复：user 子消息不是可切换分支，不能混进列表。"""
        _patch_artifacts(monkeypatch)
        parent = _msg(1, 10, None, "user", content="原始问题")
        reply = _msg(2, 10, 1, "assistant", content="回复")
        regen = _msg(3, 10, 1, "assistant", content="重新生成")
        follow_up = _msg(4, 10, 1, "user", content="追问")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(make_conv(id=10, user_id=1)),
            ai_message_repository=(_MsgRepo().put(parent).put(reply).put(regen).put(follow_up)),
        )

        result = await svc.get_branches(AsyncSession(), 10, 1, 1)
        assert [m.id for m in result] == [2, 3]

    async def test_get_branches_rejects_cross_conv_message(self):
        msg = _msg(99, 999, None, "user")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(make_conv(id=10, user_id=1)),
            ai_message_repository=_MsgRepo().put(msg),
        )
        with pytest.raises(BusinessException):
            await svc.get_branches(AsyncSession(), 10, 1, 99)

    async def test_switch_branch_updates_pointer(self):
        conv = make_conv(
            id=10,
            user_id=1,
            title="会话",
            model="gpt",
            agent_code=None,
            agent_version=None,
            summary=None,
            system_prompt=None,
            model_config=None,
            api_key_id=None,
            message_count=5,
            last_message_at=None,
            current_branch_message_id=2,
            last_read_message_id=None,
            pinned=0,
            pinned_at=None,
            delete_time=None,
            unread_count=0,
            title_source="auto",
            status=1,
            create_time=None,
            update_time=None,
        )
        target = _msg(4, 10, 1, "assistant", content="历史分支")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(conv),
            ai_message_repository=_MsgRepo().put(target),
        )

        result = await svc.switch_branch(AsyncSession(), 10, 1, 4)
        assert conv.current_branch_message_id == 4
        assert result.current_branch_message_id == 4

    async def test_switch_branch_rejected_when_suspended(self):
        """挂起期间拒绝切换：指针移走后中断入口失效，须先确认或停止。"""
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        target = _msg(4, 10, 1, "assistant", content="历史分支")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(conv),
            ai_message_repository=_MsgRepo().put(target),
            interrupt_handler=SimpleNamespace(
                get_interrupt=AsyncMock(return_value={"type": "confirm"})
            ),
        )

        with pytest.raises(BusinessException) as ei:
            await svc.switch_branch(AsyncSession(), 10, 1, 4)
        assert ei.value.code == ResultCode.BUSINESS_ERROR
        assert "中断确认" in str(ei.value.message)
        assert conv.current_branch_message_id == 2

    async def test_switch_branch_rejected_by_db_pending_without_redis(self):
        """Redis 句柄丢失不影响判定：DB pending_interrupt 仍是事实源，挂起期间拒绝切换。"""
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        tail = _msg(
            2,
            10,
            1,
            "assistant",
            content="待确认",
            status="awaiting_input",
            pending_interrupt=PENDING,
        )
        target = _msg(4, 10, 1, "assistant", content="历史分支")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(conv),
            ai_message_repository=_MsgRepo().put(tail).put(target),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
        )

        with pytest.raises(BusinessException) as ei:
            await svc.switch_branch(AsyncSession(), 10, 1, 4)
        assert ei.value.code == ResultCode.BUSINESS_ERROR
        assert "中断确认" in str(ei.value.message)
        assert conv.current_branch_message_id == 2

    async def test_switch_branch_allowed_when_db_interrupt_expired(self):
        """DB 中断已过期（actionable=false）不构成挂起，允许切换。"""
        conv = _switch_conv(id=10, user_id=1, status=1, current_branch_message_id=2)
        expired = {**PENDING, "expiresAt": "2020-01-01T00:00:00"}
        tail = _msg(2, 10, 1, "assistant", status="awaiting_input", pending_interrupt=expired)
        target = _msg(4, 10, 1, "assistant", content="历史分支")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(conv),
            ai_message_repository=_MsgRepo().put(tail).put(target),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
        )

        result = await svc.switch_branch(AsyncSession(), 10, 1, 4)
        assert conv.current_branch_message_id == 4
        assert result.current_branch_message_id == 4

    async def test_switch_branch_allowed_when_tail_not_awaiting_input(self):
        """消息已非 awaiting_input（中断已处理完）：遗留 pending 不阻止切换。"""
        conv = _switch_conv(id=10, user_id=1, status=1, current_branch_message_id=2)
        tail = _msg(2, 10, 1, "assistant", status="completed", pending_interrupt=PENDING)
        target = _msg(4, 10, 1, "assistant", content="历史分支")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(conv),
            ai_message_repository=_MsgRepo().put(tail).put(target),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
        )

        result = await svc.switch_branch(AsyncSession(), 10, 1, 4)
        assert conv.current_branch_message_id == 4
        assert result.current_branch_message_id == 4


class TestDeleteMessage:
    @staticmethod
    async def _seed_chain(db, **conv_overrides):
        """落库 u0 → a1 → u2 → a3 的激活链，指针停在 a3。"""
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt", agent_code=None, **conv_overrides)
        db.add(conv)
        await db.flush()
        msgs = []
        turns = (("user", "u0"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a3"))
        for role, text in turns:
            msg = SysAiMessage(
                conversation_id=conv.id,
                parent_message_id=msgs[-1].id if msgs else None,
                role=role,
                status="completed",
                blocks=[{"id": f"b{len(msgs)}", "kind": "text", "seq": 1, "text": text}],
            )
            db.add(msg)
            await db.flush()
            msgs.append(msg)
        conv.current_branch_message_id = msgs[-1].id
        await db.flush()
        return conv, msgs

    async def test_delete_assistant_message(self):
        msg = _msg(5, 10, 3, "assistant")
        conv = make_conv(id=10, user_id=1, current_branch_message_id=9)
        repo = _MsgRepo().put(msg)
        svc = _conv_service(ai_message_repository=repo, ai_conversation_repository=_ConvRepo(conv))
        await svc.delete_message(AsyncSession(), 5, 1)
        assert msg.deleted != 0
        # 非指针消息删除后指针不动
        assert conv.current_branch_message_id == 9

    async def test_delete_rejects_user_message(self):
        msg = _msg(5, 10, 3, "user")
        svc = _conv_service(ai_message_repository=_MsgRepo().put(msg))
        with pytest.raises(BusinessException):
            await svc.delete_message(AsyncSession(), 5, 1)

    async def test_delete_non_leaf_rejected(self, db):
        """中间回复仍有未删子消息：拒绝删除，不得断链或桥接（user/assistant 交替）。"""
        conv, (u0, a1, u2, a3) = await self._seed_chain(db)

        with pytest.raises(BusinessException) as ei:
            await ai_conversation_service.delete_message(db, a1.id, 1)

        assert ei.value.code == ResultCode.DATA_STATE_NOT_ALLOW
        assert "请先删除该回复之后的对话" in str(ei.value.message)
        # 消息仍在、指针未动，链保持完整
        await db.refresh(a1)
        assert a1.deleted == 0
        await db.refresh(conv)
        assert conv.current_branch_message_id == a3.id
        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, a3.id)
        assert [m.id for m in chain] == [u0.id, a1.id, u2.id, a3.id]

    async def test_delete_off_branch_message_keeps_pointer(self, db):
        """非激活分支上的旧回复无子消息，允许删除且指针不动。"""
        conv, (u0, a1, u2, a3) = await self._seed_chain(db)
        from app.models.entity.sys_ai_message import SysAiMessage

        old_branch = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=u0.id,
            role="assistant",
            status="completed",
            blocks=[{"id": "old", "kind": "text", "seq": 1, "text": "旧分支"}],
        )
        db.add(old_branch)
        await db.flush()

        await ai_conversation_service.delete_message(db, old_branch.id, 1)

        await db.refresh(conv)
        assert conv.current_branch_message_id == a3.id
        assert await ai_msg_repo.get_by_id(db, old_branch.id) is None
        chain = await ai_msg_repo.get_chain_by_id(db, conv.id, a3.id)
        assert [m.id for m in chain] == [u0.id, a1.id, u2.id, a3.id]

    async def test_delete_tail_moves_pointer_and_following_send_keeps_history(
        self, db, monkeypatch, mock_redis
    ):
        """删除尾部回复回退指针，继续发送后激活链仍包含此前全部历史。"""
        conv, (u0, a1, u2, a3) = await self._seed_chain(
            db,
            title="已有标题",
            title_source="auto",
            message_count=4,
            model_config={"needTools": False},
        )

        await ai_conversation_service.delete_message(db, a3.id, 1)

        await db.refresh(conv)
        assert conv.current_branch_message_id == u2.id
        assert await ai_msg_repo.get_by_id(db, a3.id) is None

        # 用真实 send_message 的落库链路继续发送（仅替换模型校验与推理流）
        async def _noop(*_args, **_kwargs):
            return None

        def _noop_sync(*_args, **_kwargs):
            return None

        async def _model(_db, _model_id):
            return SimpleNamespace(status=1)

        async def _get_redis():
            return mock_redis

        async def _fake_stream(**_kwargs):
            yield b"data"

        monkeypatch.setattr("app.service.ai_message_service.assert_message_fits_model", _noop_sync)
        monkeypatch.setattr(ai_message_service.ai_model_repository, "get_by_model_id", _model)
        monkeypatch.setattr(ai_message_service.ai_model_service, "validate_model_caps", _noop)
        monkeypatch.setattr(ai_message_service, "get_redis_client", _get_redis)
        monkeypatch.setattr("app.service.ai_message_service.stream_generator", _fake_stream)

        resp = await ai_message_service.send_message(
            db, conv.id, 1, SimpleNamespace(content="继续", model=None), "key-tail-delete"
        )
        async for _ in resp.body_iterator:
            pass

        result = await ai_conversation_service.list_messages(db, conv.id, 1, None, 50)
        chain_ids = [m.id for m in result.list]
        new_user_id, new_assistant_id = chain_ids[1], chain_ids[0]
        # 新消息挂在回退后的指针（u2）上，此前历史完整保留
        assert chain_ids[2:] == [u2.id, a1.id, u0.id]
        assert result.total == 5
        sent = await ai_msg_repo.get_by_id(db, new_user_id)
        assert sent.parent_message_id == u2.id
        assert new_assistant_id > new_user_id

    async def test_delete_root_tail_clears_pointer(self, db):
        """root 级助手消息（父为空）删除后指针置空，链按最大未删消息回退。"""
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt")
        db.add(conv)
        await db.flush()
        root_asst = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=None,
            role="assistant",
            status="completed",
            blocks=[],
        )
        db.add(root_asst)
        await db.flush()
        conv.current_branch_message_id = root_asst.id
        await db.flush()

        await ai_conversation_service.delete_message(db, root_asst.id, 1)

        await db.refresh(conv)
        assert conv.current_branch_message_id is None

    async def test_delete_locks_conversation_row(self, db):
        """删除在会话行锁内进行：并发 send 的 update_last_message 与本删除天然串行。"""
        from sqlalchemy import event as sa_event

        conv, (_, _, u2, a3) = await self._seed_chain(db)
        statements = []

        def _record(state):
            statements.append(str(state.statement))

        sa_event.listen(db.sync_session, "do_orm_execute", _record)
        try:
            await ai_conversation_service.delete_message(db, a3.id, 1)
        finally:
            sa_event.remove(db.sync_session, "do_orm_execute", _record)

        # 删除必须真正对会话行加锁（repository 支持 for_update 不代表调用方用了）
        assert any("sys_ai_conversation" in sql and "FOR UPDATE" in sql for sql in statements), (
            statements
        )
        await db.refresh(conv)
        assert conv.current_branch_message_id == u2.id

    async def test_delete_tail_does_not_rollback_pointer_advanced_by_send(self, db):
        """并发 send 已推进指针后再删旧尾：不得把指针覆盖回旧父。

        真实并发在单连接测试夹具内无法复现，用 set_committed_value 把会话对象还原成
        "本请求加锁前读到的旧快照"——旧实现会据此无条件回写，覆盖另一事务的新指针。
        """
        from sqlalchemy.orm.attributes import set_committed_value

        from app.models.entity.sys_ai_message import SysAiMessage

        conv, (_, _, u2, a3) = await self._seed_chain(db)
        new_asst = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=u2.id,
            role="assistant",
            status="completed",
            blocks=[],
        )
        db.add(new_asst)
        await db.flush()
        # 并发 send 落库并推进指针（另一事务已提交）
        await ai_conversation_repository.update_last_message(
            db, conv.id, new_asst.id, datetime.now()
        )
        # 本请求此前读到的指针快照仍是 a3
        set_committed_value(conv, "current_branch_message_id", a3.id)

        await ai_conversation_service.delete_message(db, a3.id, 1)

        await db.refresh(conv)
        assert await ai_msg_repo.get_by_id(db, a3.id) is None
        assert conv.current_branch_message_id == new_asst.id

    async def test_pointer_cas_ignores_stale_expectation(self, db):
        """CAS 回写：期望值不符时不写入，过期快照不会把指针拽回已删节点。"""
        conv, (_, _, u2, a3) = await self._seed_chain(db)

        moved = await ai_conversation_repository.update_current_branch_if(
            db, conv.id, expected_message_id=a3.id, message_id=u2.id
        )
        assert moved is True
        await db.refresh(conv)
        assert conv.current_branch_message_id == u2.id

        # 过期快照（仍以为指针是 a3）再次回写：不命中，指针保持 u2
        stale = await ai_conversation_repository.update_current_branch_if(
            db, conv.id, expected_message_id=a3.id, message_id=None
        )
        assert stale is False
        await db.refresh(conv)
        assert conv.current_branch_message_id == u2.id


class TestSuspendLock:
    @staticmethod
    async def _seed_run_ctx(db):
        """落库真实会话 + 用户消息 + 待生成的 assistant 消息（status=running）"""
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt")
        db.add(conv)
        await db.flush()
        user_msg = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=None,
            role="user",
            status="completed",
            blocks=[{"id": "u1", "kind": "text", "seq": 1, "text": "你好"}],
        )
        db.add(user_msg)
        await db.flush()
        asst_msg = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=user_msg.id,
            role="assistant",
            status="running",
            blocks=[],
        )
        db.add(asst_msg)
        await db.flush()
        conv.current_branch_message_id = asst_msg.id
        await db.flush()
        return conv, asst_msg

    async def test_run_releases_lock_on_suspend(self, db, monkeypatch):
        conv, asst_msg = await self._seed_run_ctx(db)
        service, emitter, _ = patch_reasoning_boundaries(
            monkeypatch, interrupt={"type": "confirm", "data": {"stream_session_id": "s1"}}
        )
        await service.run(
            conv_id=conv.id, user_id=1, msg_id=asst_msg.id, model_id="gpt", stream_session_id="s1"
        )
        # 挂起：让渡会话并发锁给 resume（release_lock 幂等，可安全让渡）
        assert emitter.released == [conv.id]
        # message.end 收尾作为本轮流结束信号
        assert emitter.events[-1][0] == "message.end"

    async def test_run_keeps_lock_without_suspend(self, db, monkeypatch):
        conv, asst_msg = await self._seed_run_ctx(db)
        service, emitter, _ = patch_reasoning_boundaries(monkeypatch, interrupt=None)
        await service.run(
            conv_id=conv.id, user_id=1, msg_id=asst_msg.id, model_id="gpt", stream_session_id="s1"
        )
        # 未挂起：不释放锁
        assert emitter.released == []
        # 且整条链路真实完成落库（业务结果）
        await db.refresh(asst_msg)
        assert asst_msg.status == "completed"
        assert asst_msg.text == "ok"
        assert emitter.events[-1][0] == "message.end"

    async def test_send_rejected_when_suspended(self, mock_redis):
        conv = make_conv(
            id=10,
            user_id=1,
            status=1,
            current_branch_message_id=5,
            model="gpt",
            model_config=None,
            agent_code=None,
        )
        svc = AiMessageService(
            # 替身：_ConvRepo 返回 SimpleNamespace 会话（非实体），子类化会触发方法覆写告警
            ai_conversation_repository=cast(AiConversationRepository, _ConvRepo(conv)),
            interrupt_handler=_InterruptStub({"type": "confirm"}),
            get_redis_client=lambda: mock_redis,
        )

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException) as ei:
            await svc.send_message(AsyncSession(), 10, 1, form, "k")
        assert "中断确认" in str(ei.value.message)

    async def test_send_allowed_when_no_suspend(self, mock_redis):
        conv = make_conv(
            id=10,
            user_id=1,
            status=1,
            current_branch_message_id=5,
            model="gpt",
            model_config=None,
            agent_code=None,
            message_count=2,
            title="会话",
            title_source="auto",
        )
        lock_called = []

        async def fake_acquire(cid):
            lock_called.append(cid)
            return False

        svc = AiMessageService(
            # 替身：_ConvRepo 返回 SimpleNamespace 会话（非实体），子类化会触发方法覆写告警
            ai_conversation_repository=cast(AiConversationRepository, _ConvRepo(conv)),
            interrupt_handler=_InterruptStub(None),
            get_redis_client=lambda: mock_redis,
            sse_emitter_manager=_EmitterStub(acquire_lock=fake_acquire),
        )

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException):
            await svc.send_message(AsyncSession(), 10, 1, form, "k")
        assert lock_called == [10]

    async def test_regenerate_rejected_when_suspended(self):
        msg = _msg(2, 10, 1, "assistant", status="completed", task_id=None)
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            ai_conversation_repository=_ConvRepo(conv),
            interrupt_handler=SimpleNamespace(
                get_interrupt=AsyncMock(return_value={"type": "confirm"})
            ),
        )

        with pytest.raises(BusinessException) as ei:
            await svc.regenerate_message(AsyncSession(), 2, 1)
        assert "中断确认" in str(ei.value.message)
