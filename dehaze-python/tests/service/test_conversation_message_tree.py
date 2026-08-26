from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.requires_db

from app.core.exceptions import BusinessException
from app.models.schema.ai_conversation import MessageResume
from app.repository.ai_message_repository import ai_message_repository as ai_msg_repo
from app.service.ai_conversation_service import AiConversationService
from app.service.ai import message_streaming
from app.service.ai_message_service import AiMessageService
from app.service.ai.builders.context_manager import context_manager
from tests.stubs.factories import make_conv
from tests.stubs.mocks import patch_reasoning_boundaries


def _msg(id, conv_id, parent, role, **kw):
    base = dict(
        id=id,
        conversation_id=conv_id,
        parent_message_id=parent,
        role=role,
        deleted=0,
        content="",
        model="gpt",
        status=2,
        task_id=None,
        error=None,
        tool_calls=None,
        tool_call_id=None,
        metadata_=None,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        credits=0,
        edited=0,
        original_content=None,
        create_time=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


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

    async def create(self, db, entity):
        entity.id = self._next_id
        self._next_id += 1
        self.msgs[entity.id] = entity
        self.created.append(entity)
        return entity

    async def get_children(self, db, cid, pid):
        return [
            m
            for m in self.msgs.values()
            if m.conversation_id == cid and m.parent_message_id == pid
        ]

    async def soft_delete_by_ids(self, db, ids):
        for i in ids:
            self.msgs[i].deleted = 1

    async def update_status(self, db, mid, status):
        self.msgs[mid].status = status


class _ConvRepo:
    def __init__(self, conv=None):
        self._conv = conv

    async def get_by_id_and_user(self, db, cid, uid):
        return self._conv

    async def update_last_message(self, db, cid, mid, t, **kw):
        if self._conv is not None:
            self._conv.current_branch_message_id = mid

    async def update_current_branch(self, db, cid, mid):
        if self._conv is not None:
            self._conv.current_branch_message_id = mid


def _conv_service(**kw):
    """构造 AiConversationService：未显式注入的依赖回落模块级单例（测试中仅用桩字段）"""
    return AiConversationService(**kw)


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

        resp = await svc.regenerate_message(object(), 2, 1)
        assert svc.ai_message_repository.created[0].role == "assistant"
        assert svc.ai_message_repository.created[0].parent_message_id == 1
        assert svc.ai_message_repository.created[0].id == 3
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
            await svc.regenerate_message(object(), 2, 1)

    async def test_shared_reasoning_trigger(self, mock_redis):
        run_calls = {}

        async def fake_run(**kw):
            run_calls.update(kw)

        async def fake_stop(sid):
            pass

        svc = AiMessageService(
            reasoning_service=SimpleNamespace(run=fake_run),
            sse_emitter_manager=SimpleNamespace(stop_stream=fake_stop),
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
    async def test_no_cross_branch_contamination(self, monkeypatch):
        msgs = {
            1: _msg(1, 10, None, "user", content="a"),
            2: _msg(2, 10, 1, "assistant", content="branchA"),
            3: _msg(3, 10, 1, "user", content="b"),
            4: _msg(4, 10, 3, "assistant", content="branchB"),
        }

        async def get_by_ids(db, ids):
            return [msgs[i] for i in ids if i in msgs]

        monkeypatch.setattr(ai_msg_repo, "get_by_ids", get_by_ids)
        chain = await ai_msg_repo.get_chain_by_id(object(), 10, 4)
        assert [m.id for m in chain] == [1, 3, 4]

    async def test_ring_protection(self, monkeypatch):
        m1 = _msg(1, 10, 2, "user", content="x")
        m2 = _msg(2, 10, 1, "assistant", content="y")

        async def get_by_ids(db, ids):
            return [m1] if ids[0] == 1 else [m2]

        monkeypatch.setattr(ai_msg_repo, "get_by_ids", get_by_ids)
        chain = await ai_msg_repo.get_chain_by_id(object(), 10, 1)
        assert [m.id for m in chain] == [2, 1]

    async def test_limit_respected(self, monkeypatch):
        msgs = {i: _msg(i, 10, i - 1, "user", content=str(i)) for i in range(1, 50)}

        async def get_by_ids(db, ids):
            return [msgs[i] for i in ids if i in msgs]

        monkeypatch.setattr(ai_msg_repo, "get_by_ids", get_by_ids)
        chain = await ai_msg_repo.get_chain_by_id(object(), 10, 49, limit=20)
        assert len(chain) == 20
        assert chain[0].id == 30 and chain[-1].id == 49

    async def test_context_load_uses_branch_chain(self, monkeypatch):
        conv = make_conv(id=10, user_id=1, current_branch_message_id=4)
        msgs = {
            1: _msg(1, 10, None, "user", content="a"),
            2: _msg(2, 10, 1, "assistant", content="branchA"),
            3: _msg(3, 10, 1, "user", content="b"),
            4: _msg(4, 10, 3, "assistant", content="branchB"),
        }

        async def get_by_ids(db, ids):
            return [msgs[i] for i in ids if i in msgs]

        monkeypatch.setattr(
            "app.service.ai.builders.context_manager.inject_memories", AsyncMock(return_value=(None, []))
        )
        monkeypatch.setattr(ai_msg_repo, "get_by_ids", get_by_ids)
        messages, _system_prompt, _injected = await context_manager.build_context(
            object(), conv, "gpt"
        )
        assert [{k: v for k, v in m.items() if k != "id"} for m in messages] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "branchB"},
        ]
        assert all(m["content"] != "branchA" for m in messages)


class TestResumeStop:
    async def test_resume_triggers_reasoning(self):
        msg = _msg(5, 10, 3, "assistant", status=1)
        resume_calls = []

        async def get_interrupt(tid):
            assert tid == "10:5"
            return {"type": "confirm", "data": {"stream_session_id": "s1"}}

        async def fake_resume(conv_id, user_id, msg_id, resume_data):
            resume_calls.append((conv_id, user_id, msg_id, resume_data))

        async def fake_create_stream(cid, sid):
            yield b"data: 1"

        async def fake_stop_stream(sid):
            return None

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            interrupt_handler=SimpleNamespace(get_interrupt=get_interrupt),
            sse_emitter_manager=SimpleNamespace(
                acquire_lock=AsyncMock(return_value=True),
                create_stream=fake_create_stream,
                stop_stream=fake_stop_stream,
            ),
            reasoning_service=SimpleNamespace(resume=fake_resume),
        )

        form = MessageResume(confirm=True, params={"algorithmId": 7})
        resp = await svc.resume_message(object(), 5, 1, form)
        async for _ in resp.body_iterator:
            pass
        assert resp.media_type == "text/event-stream"
        assert resume_calls and resume_calls[0][0:3] == (10, 1, 5)
        assert resume_calls[0][3] == {"confirmed": True, "algorithmId": 7}

    async def test_resume_no_interrupt(self):
        msg = _msg(5, 10, 3, "assistant")
        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
        )
        with pytest.raises(BusinessException):
            await svc.resume_message(object(), 5, 1, MessageResume(confirm=False))

    async def test_stop_calls_reasoning_stop(self):
        msg = _msg(5, 10, 3, "assistant", status=1, task_id="s1")
        stop_calls = []

        async def fake_stop(conv_id, msg_id, sid):
            stop_calls.append((conv_id, msg_id, sid))

        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            reasoning_service=SimpleNamespace(stop=fake_stop),
        )
        result = await svc.stop_message(object(), 5, 1)
        assert stop_calls == [(10, 5, "s1")]
        assert result.status == 4


class TestBranches:
    async def test_get_branches_lists_children(self):
        children = [
            _msg(2, 10, 1, "assistant", content="主分支"),
            _msg(3, 10, 1, "assistant", content="regenerate分支"),
        ]
        msg = _msg(1, 10, None, "user", content="原始问题")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(make_conv(id=10, user_id=1)),
            ai_message_repository=_MsgRepo().put(msg).put(children[0]).put(children[1]),
        )

        result = await svc.get_branches(object(), 10, 1, 1)
        assert [m.id for m in result] == [2, 3]

    async def test_get_branches_rejects_cross_conv_message(self):
        msg = _msg(99, 999, None, "user")
        svc = _conv_service(
            ai_conversation_repository=_ConvRepo(make_conv(id=10, user_id=1)),
            ai_message_repository=_MsgRepo().put(msg),
        )
        with pytest.raises(BusinessException):
            await svc.get_branches(object(), 10, 1, 99)

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

        result = await svc.switch_branch(object(), 10, 1, 4)
        assert conv.current_branch_message_id == 4
        assert result.current_branch_message_id == 4


class TestDeleteMessage:
    async def test_delete_assistant_message(self):
        msg = _msg(5, 10, 3, "assistant")
        repo = _MsgRepo().put(msg)
        svc = _conv_service(ai_message_repository=repo)
        await svc.delete_message(object(), 5, 1)
        assert msg.deleted == 1

    async def test_delete_rejects_user_message(self):
        msg = _msg(5, 10, 3, "user")
        svc = _conv_service(ai_message_repository=_MsgRepo().put(msg))
        with pytest.raises(BusinessException):
            await svc.delete_message(object(), 5, 1)


class TestSuspendLock:
    @staticmethod
    async def _seed_run_ctx(db):
        """落库真实会话 + 用户消息 + 待生成的 assistant 消息（status=1 生成中）"""
        from app.models.entity.sys_ai_conversation import SysAiConversation
        from app.models.entity.sys_ai_message import SysAiMessage

        conv = SysAiConversation(user_id=1, model="gpt")
        db.add(conv)
        await db.flush()
        user_msg = SysAiMessage(
            conversation_id=conv.id, parent_message_id=None, role="user", content="你好", status=2
        )
        db.add(user_msg)
        await db.flush()
        asst_msg = SysAiMessage(
            conversation_id=conv.id,
            parent_message_id=user_msg.id,
            role="assistant",
            content="",
            status=1,
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
        assert asst_msg.status == 2
        assert asst_msg.content == "ok"
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
            ai_conversation_repository=_ConvRepo(conv),
            interrupt_handler=SimpleNamespace(
                get_interrupt=AsyncMock(return_value={"type": "confirm"})
            ),
            get_redis_client=lambda: mock_redis,
        )

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException) as ei:
            await svc.send_message(object(), 10, 1, form, "k")
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
            ai_conversation_repository=_ConvRepo(conv),
            interrupt_handler=SimpleNamespace(get_interrupt=AsyncMock(return_value=None)),
            get_redis_client=lambda: mock_redis,
            sse_emitter_manager=SimpleNamespace(acquire_lock=fake_acquire),
        )

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException):
            await svc.send_message(object(), 10, 1, form, "k")
        assert lock_called == [10]

    async def test_regenerate_rejected_when_suspended(self):
        msg = _msg(2, 10, 1, "assistant", status=2, task_id=None)
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        svc = _conv_service(
            ai_message_repository=_MsgRepo().put(msg),
            ai_conversation_repository=_ConvRepo(conv),
            interrupt_handler=SimpleNamespace(
                get_interrupt=AsyncMock(return_value={"type": "confirm"})
            ),
        )

        with pytest.raises(BusinessException) as ei:
            await svc.regenerate_message(object(), 2, 1)
        assert "中断确认" in str(ei.value.message)
