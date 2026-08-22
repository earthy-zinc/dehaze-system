from types import SimpleNamespace

import pytest

import app.service.ai_conversation_service as svc
import app.service.ai_message_service as msg_svc
from app.core.exceptions import BusinessException
from app.models.schema.ai_conversation import MessageResume
from app.service.ai.context_manager import context_manager
from tests.stubs import async_ret, install_reasoning_chain_mocks, make_conv


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


class TestRegenerate:
    async def test_branch_structure_and_wiring(self, monkeypatch):
        user_msg = _msg(1, 10, None, "user", content="你好")
        old_asst = _msg(2, 10, 1, "assistant", content="旧回复")
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        created = {}
        wired = {}

        async def get_by_id_and_user(db, mid, uid):
            return {1: user_msg, 2: old_asst}.get(mid)

        async def create(db, entity):
            entity.id = 3
            created["msg"] = entity
            return entity

        async def update_last_message(db, cid, mid, t):
            conv.current_branch_message_id = mid

        async def fake_stream(**kw):
            wired["assistant_msg_id"] = kw["assistant_msg_id"]
            wired["stream_session_id"] = kw["stream_session_id"]
            yield b"data"

        monkeypatch.setattr(svc.ai_message_repository, "get_by_id_and_user", get_by_id_and_user)
        monkeypatch.setattr(
            svc.ai_conversation_repository, "get_by_id_and_user", async_ret(conv)
        )
        monkeypatch.setattr(svc.ai_message_repository, "create", create)
        monkeypatch.setattr(
            svc.ai_conversation_repository, "update_last_message", update_last_message
        )
        monkeypatch.setattr(svc.sse_emitter_manager, "acquire_lock", async_ret(True))
        monkeypatch.setattr(svc.interrupt_handler, "get_interrupt", async_ret(None))
        monkeypatch.setattr("app.service.ai_message_service._stream_generator", fake_stream)

        resp = await svc.ai_conversation_service.regenerate_message(object(), 2, 1)
        assert created["msg"].role == "assistant"
        assert created["msg"].parent_message_id == 1
        assert created["msg"].id == 3
        assert conv.current_branch_message_id == 3
        assert resp.media_type == "text/event-stream"
        async for _ in resp.body_iterator:
            pass
        assert wired["assistant_msg_id"] == 3

    async def test_rejects_deleted_message(self, monkeypatch):
        deleted_asst = _msg(2, 10, 1, "assistant", deleted=1)
        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(deleted_asst)
        )
        with pytest.raises(BusinessException):
            await svc.ai_conversation_service.regenerate_message(object(), 2, 1)

    async def test_shared_reasoning_trigger(self, monkeypatch, mock_redis):
        run_calls = {}

        async def fake_run(**kw):
            run_calls.update(kw)

        monkeypatch.setattr(msg_svc.reasoning_service, "run", fake_run)
        monkeypatch.setattr(msg_svc, "get_redis_client", async_ret(mock_redis))
        stopped = []

        async def fake_stop(sid):
            stopped.append(sid)

        monkeypatch.setattr(msg_svc.sse_emitter_manager, "stop_stream", fake_stop)

        await msg_svc._run_reasoning(
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
        assert stopped == ["s1"]


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

        monkeypatch.setattr(svc.ai_message_repository, "get_by_ids", get_by_ids)
        chain = await svc.ai_message_repository.get_chain_by_id(object(), 10, 4)
        assert [m.id for m in chain] == [1, 3, 4]

    async def test_ring_protection(self, monkeypatch):
        m1 = _msg(1, 10, 2, "user", content="x")
        m2 = _msg(2, 10, 1, "assistant", content="y")

        async def get_by_ids(db, ids):
            return [m1] if ids[0] == 1 else [m2]

        monkeypatch.setattr(svc.ai_message_repository, "get_by_ids", get_by_ids)
        chain = await svc.ai_message_repository.get_chain_by_id(object(), 10, 1)
        assert [m.id for m in chain] == [2, 1]

    async def test_limit_respected(self, monkeypatch):
        msgs = {i: _msg(i, 10, i - 1, "user", content=str(i)) for i in range(1, 50)}

        async def get_by_ids(db, ids):
            return [msgs[i] for i in ids if i in msgs]

        monkeypatch.setattr(svc.ai_message_repository, "get_by_ids", get_by_ids)
        chain = await svc.ai_message_repository.get_chain_by_id(object(), 10, 49, limit=20)
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
            "app.service.ai.context_manager.inject_memories", async_ret((None, []))
        )
        monkeypatch.setattr(svc.ai_message_repository, "get_by_ids", get_by_ids)
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
    async def test_resume_triggers_reasoning(self, monkeypatch):
        msg = _msg(5, 10, 3, "assistant", status=1)
        resume_calls = []

        async def get_interrupt(tid):
            assert tid == "10:5"
            return {"type": "confirm", "data": {"stream_session_id": "s1"}}

        async def fake_resume(conv_id, user_id, msg_id, resume_data):
            resume_calls.append((conv_id, user_id, msg_id, resume_data))

        async def fake_create_stream(cid, sid):
            yield b"data: 1"

        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(msg)
        )
        monkeypatch.setattr(svc.interrupt_handler, "get_interrupt", get_interrupt)
        monkeypatch.setattr(svc.sse_emitter_manager, "acquire_lock", async_ret(True))
        monkeypatch.setattr(svc.reasoning_service, "resume", fake_resume)
        monkeypatch.setattr(svc.sse_emitter_manager, "create_stream", fake_create_stream)
        monkeypatch.setattr(svc.sse_emitter_manager, "stop_stream", async_ret(None))

        form = MessageResume(confirm=True, params={"algorithmId": 7})
        resp = await svc.ai_conversation_service.resume_message(object(), 5, 1, form)
        async for _ in resp.body_iterator:
            pass
        assert resp.media_type == "text/event-stream"
        assert resume_calls and resume_calls[0][0:3] == (10, 1, 5)
        assert resume_calls[0][3] == {"confirmed": True, "algorithmId": 7}

    async def test_resume_no_interrupt(self, monkeypatch):
        msg = _msg(5, 10, 3, "assistant")
        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(msg)
        )
        monkeypatch.setattr(svc.interrupt_handler, "get_interrupt", async_ret(None))
        with pytest.raises(BusinessException):
            await svc.ai_conversation_service.resume_message(
                object(), 5, 1, MessageResume(confirm=False)
            )

    async def test_stop_calls_reasoning_stop(self, monkeypatch):
        msg = _msg(5, 10, 3, "assistant", status=1, task_id="s1")
        stop_calls = []

        async def fake_stop(conv_id, msg_id, sid):
            stop_calls.append((conv_id, msg_id, sid))

        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(msg)
        )
        monkeypatch.setattr(svc.reasoning_service, "stop", fake_stop)
        result = await svc.ai_conversation_service.stop_message(object(), 5, 1)
        assert stop_calls == [(10, 5, "s1")]
        assert result.status == 4


class TestBranches:
    async def test_get_branches_lists_children(self, monkeypatch):
        children = [
            _msg(2, 10, 1, "assistant", content="主分支"),
            _msg(3, 10, 1, "assistant", content="regenerate分支"),
        ]
        msg = _msg(1, 10, None, "user", content="原始问题")

        async def get_children(db, cid, pid):
            assert (cid, pid) == (10, 1)
            return children

        monkeypatch.setattr(
            svc.ai_conversation_repository, "get_by_id_and_user",
            async_ret(make_conv(id=10, user_id=1)),
        )
        monkeypatch.setattr(svc.ai_message_repository, "get_by_id_and_user", async_ret(msg))
        monkeypatch.setattr(svc.ai_message_repository, "get_children", get_children)

        result = await svc.ai_conversation_service.get_branches(object(), 10, 1, 1)
        assert [m.id for m in result] == [2, 3]

    async def test_get_branches_rejects_cross_conv_message(self, monkeypatch):
        msg = _msg(99, 999, None, "user")
        monkeypatch.setattr(
            svc.ai_conversation_repository, "get_by_id_and_user",
            async_ret(make_conv(id=10, user_id=1)),
        )
        monkeypatch.setattr(svc.ai_message_repository, "get_by_id_and_user", async_ret(msg))
        with pytest.raises(BusinessException):
            await svc.ai_conversation_service.get_branches(object(), 10, 1, 99)

    async def test_switch_branch_updates_pointer(self, monkeypatch):
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
        updated = {}

        async def update_branch(db, cid, mid):
            updated["cid"] = cid
            updated["mid"] = mid

        monkeypatch.setattr(
            svc.ai_conversation_repository, "get_by_id_and_user", async_ret(conv)
        )
        monkeypatch.setattr(svc.ai_message_repository, "get_by_id_and_user", async_ret(target))
        monkeypatch.setattr(svc.ai_conversation_repository, "update_current_branch", update_branch)

        result = await svc.ai_conversation_service.switch_branch(object(), 10, 1, 4)
        assert updated == {"cid": 10, "mid": 4}
        assert result.current_branch_message_id == 4


class TestDeleteMessage:
    async def test_delete_assistant_message(self, monkeypatch):
        msg = _msg(5, 10, 3, "assistant")
        deleted = []

        async def soft_delete(db, ids):
            deleted.extend(ids)

        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(msg)
        )
        monkeypatch.setattr(svc.ai_message_repository, "soft_delete_by_ids", soft_delete)
        await svc.ai_conversation_service.delete_message(object(), 5, 1)
        assert deleted == [5]

    async def test_delete_rejects_user_message(self, monkeypatch):
        msg = _msg(5, 10, 3, "user")
        monkeypatch.setattr(
            svc.ai_message_repository, "get_by_id_and_user", async_ret(msg)
        )
        with pytest.raises(BusinessException):
            await svc.ai_conversation_service.delete_message(object(), 5, 1)


class TestSuspendLock:
    async def test_run_releases_lock_on_suspend(self, monkeypatch):
        _service, recorder = install_reasoning_chain_mocks(
            monkeypatch,
            interrupt={"type": "confirm"},
            values={"final_response": "", "stop_reason": "interrupt"},
            conv=make_conv(id=10, user_id=1, status=1, current_branch_message_id=5, model="gpt"),
            interrupt_tid="10:5",
        )
        await _service.run(
            conv_id=10, user_id=1, msg_id=5, model_id="gpt", stream_session_id="s1"
        )
        assert recorder["released"] == [10]

    async def test_run_keeps_lock_without_suspend(self, monkeypatch):
        _service, recorder = install_reasoning_chain_mocks(
            monkeypatch,
            interrupt=None,
            values={"final_response": "ok", "stop_reason": "stop"},
            conv=make_conv(id=10, user_id=1, status=1, current_branch_message_id=5, model="gpt"),
            interrupt_tid="10:5",
        )
        await _service.run(
            conv_id=10, user_id=1, msg_id=5, model_id="gpt", stream_session_id="s1"
        )
        assert recorder["released"] == []

    async def test_send_rejected_when_suspended(self, monkeypatch):
        conv = make_conv(
            id=10,
            user_id=1,
            status=1,
            current_branch_message_id=5,
            model="gpt",
            model_config=None,
            agent_code=None,
        )
        monkeypatch.setattr(
            msg_svc.ai_conversation_repository, "get_by_id_and_user", async_ret(conv)
        )
        monkeypatch.setattr(
            msg_svc.interrupt_handler, "get_interrupt", async_ret({"type": "confirm"})
        )

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException) as ei:
            await msg_svc.ai_message_service.send_message(object(), 10, 1, form, "k")
        assert "中断确认" in str(ei.value.message)

    async def test_send_allowed_when_no_suspend(self, monkeypatch, mock_redis):
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
        monkeypatch.setattr(
            msg_svc.ai_conversation_repository, "get_by_id_and_user", async_ret(conv)
        )
        monkeypatch.setattr(msg_svc.interrupt_handler, "get_interrupt", async_ret(None))
        monkeypatch.setattr(msg_svc, "get_redis_client", async_ret(mock_redis))
        lock_called = []

        async def fake_acquire(cid):
            lock_called.append(cid)
            return False

        monkeypatch.setattr(msg_svc.sse_emitter_manager, "acquire_lock", fake_acquire)

        form = SimpleNamespace(content="hi")
        with pytest.raises(BusinessException):
            await msg_svc.ai_message_service.send_message(object(), 10, 1, form, "k")
        assert lock_called == [10]

    async def test_regenerate_rejected_when_suspended(self, monkeypatch):
        msg = _msg(2, 10, 1, "assistant", status=2, task_id=None)
        conv = make_conv(id=10, user_id=1, status=1, current_branch_message_id=2, model="gpt")
        monkeypatch.setattr(svc.ai_message_repository, "get_by_id_and_user", async_ret(msg))
        monkeypatch.setattr(
            svc.ai_conversation_repository, "get_by_id_and_user", async_ret(conv)
        )
        monkeypatch.setattr(
            svc.interrupt_handler, "get_interrupt", async_ret({"type": "confirm"})
        )

        with pytest.raises(BusinessException) as ei:
            await svc.ai_conversation_service.regenerate_message(object(), 2, 1)
        assert "中断确认" in str(ei.value.message)
