"""全域共享测试桩：真实协议 Redis、DB/图/LLM/httpx 假件与通用实体工厂。"""

import asyncio
import httpx
from datetime import datetime
from types import SimpleNamespace

from fakeredis import FakeAsyncRedis


async def fake_redis(data: dict | None = None) -> FakeAsyncRedis:
    """构造带初始数据的 fakeredis 客户端（异步工厂，测试内 await 使用）"""
    client = FakeAsyncRedis(decode_responses=True)
    for key, value in (data or {}).items():
        await client.set(key, value)
    return client


class _Savepoint:
    """begin_nested() 的 SAVEPOINT 上下文：退出时计数，异常继续外传由调用方捕获。"""

    def __init__(self, session: "StubAsyncSession"):
        self._session = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._session.savepoint_released += 1
        return False


class StubAsyncSession:
    """AsyncSession 假件：service 层单测的标准 DB 桩。

    属性即断言面：added（最后一次 add 的实体）/ entities（全部）/
    flushed / committed / rolled_back（次数）/ refreshed（实体列表）/
    savepoint_released（SAVEPOINT 释放次数）/ next_id（自增 id 发号器）。

    flush 模拟 ORM 行为：为无 id 的新实体分配自增 id（repository.create
    场景落库后实体可读 id，与真实 flush 语义一致）。
    """

    def __init__(self):
        self.added = None
        self.entities = []
        self.flushed = 0
        self.committed = 0
        self.rolled_back = 0
        self.refreshed = []
        self.savepoint_released = 0
        self.next_id = 1

    def add(self, entity):
        self.added = entity
        self.entities.append(entity)

    def add_all(self, entities):
        self.entities.extend(entities)

    async def flush(self):
        self.flushed += 1
        for entity in self.entities:
            if getattr(entity, "id", None) is None:
                entity.id = self.next_id
                self.next_id += 1

    async def commit(self):
        self.committed += 1

    async def rollback(self):
        self.rolled_back += 1

    async def refresh(self, entity):
        self.refreshed.append(entity)

    def begin_nested(self):
        return _Savepoint(self)


class NullDBSession:
    """`async with get_db_session() as db` 的 no-op 桩。

    仓储层全 mock 的场景（业务逻辑走桩、session 仅承担事务语义），
    合并各文件逐字重复的 `_DB: pass` / `_DBSession` / 带 commit 的 `_DB`。
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def flush(self):
        return None

    async def refresh(self, entity):
        return None

    async def execute(self, stmt):
        return None


class StubInterruptHandler:
    """interrupt_handler 假件。

    - interrupt：构造注入的挂起中断（get_interrupt 返回值），None 即无中断
    - saved / cleared：记录 save_interrupt/clear_interrupt 的调用参数
    """

    def __init__(self, interrupt=None):
        self.interrupt = interrupt
        self.saved = []
        self.cleared = []

    async def get_interrupt(self, thread_id):
        return self.interrupt

    async def save_interrupt(self, thread_id, itype, data):
        self.saved.append((thread_id, itype, data))

    async def clear_interrupt(self, thread_id):
        self.cleared.append(thread_id)


class RecorderEmitter:
    """SSE 发射器假件：按序捕获 send_event 调用。

    events 即断言面：[(event_type, data), ...]，按发送顺序排列。
    """

    def __init__(self):
        self.events = []

    async def send_event(self, stream_session_id, event_type, data):
        self.events.append((event_type, data))


class MinimalExecutorDB:
    """executor 所需的最小 DB 桩：commit 计数，其余 no-op（execute 返回 None）。

    与 StubAsyncSession 的差异：面向 `ScheduleExecutor.trigger_once` 的
    仓储层全 mock 场景，session 仅承担事务提交语义。
    """

    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def flush(self):
        pass

    async def execute(self, stmt):
        return None

    def add(self, obj):
        pass


def run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def async_ret(value):
    async def _stub(*a, **k):
        return value

    return _stub


def make_orm_mem(
    id_,
    type_,
    content,
    importance=50,
    metadata=None,
    last_accessed=None,
    create=None,
    status=1,
    archived=0,
    deleted=0,
    **extra,
):
    fields = {
        "id": id_,
        "memory_type": type_,
        "content": content,
        "importance": importance,
        "metadata_": metadata,
        "last_accessed_at": last_accessed,
        "create_time": create or datetime.now(),
        "status": status,
        "archived": archived,
        "deleted": deleted,
    }
    fields.update(extra)
    return type("M", (), fields)()


class LLMChunk:
    def __init__(self, type="text_delta", content="", usage=None):
        self.type = type
        self.content = content
        self.usage = usage


class FakeStreamResponse:
    def __init__(self, status_code=200, lines=()):
        self.status_code = status_code
        self._lines = list(lines)

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError(
                "call failed",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeInternalResponse:
    """内部 SSE 响应桩：body_iterator 逐条产出内联事件文本。

    统一各兼容协议测试文件重复的 body_iterator 构造（原 _InternalSSE /
    _stream_response）。
    """

    def __init__(self, *chunks):
        self.body_iterator = self._gen(chunks)

    async def _gen(self, chunks):
        for chunk in chunks:
            yield chunk


class FakeGraph:
    def __init__(self, values=None):
        self.values = values if values is not None else {}

    async def astream(self, *a, **k):
        for _ in ():
            yield _

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values)


def make_conv(**overrides):
    fields = {
        "id": 1,
        "user_id": 1,
        "system_prompt": None,
        "current_branch_message_id": None,
        "summary": None,
        "agent_code": None,
        "agent_version": None,
        "status": 1,
        "model": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_member(level_code="level_0"):
    return SimpleNamespace(level_code=level_code)


def make_benefit(multimodal_limit=0, **extra):
    fields = {"multimodal_limit": multimodal_limit}
    fields.update(extra)
    return SimpleNamespace(**fields)


class MemberBenefitRepo:
    def __init__(self, member=None, benefit=None):
        self.member = member
        self.benefit = benefit

    async def get_by_user_id(self, db, user_id):
        return self.member

    async def get_by_level_code(self, db, level_code):
        return self.benefit


def repo_returns(obj):
    """get_by_id 恒返回固定对象的仓储桩（ai_message_repository 等）。"""

    class _Repo:
        async def get_by_id(self, db, msg_id):
            return obj

    return _Repo()


def make_user_context(id=1, username="u", **overrides):
    from app.dependencies.auth import UserContext

    fields = {"id": id, "username": username}
    fields.update(overrides)
    return UserContext(**fields)


def install_reasoning_chain_mocks(
    monkeypatch,
    *,
    interrupt=None,
    values=None,
    graph=None,
    injected_list=None,
    snapshot=None,
    resolve_mode=("react", 10),
    conv=None,
    interrupt_tid=None,
):
    from app.service.ai.deep_agent_builder import DeepAgentBuilder
    from app.service.ai.reasoning_service import ReasoningService

    recorder = {
        "ctx_calls": 0,
        "finalized": [],
        "pushed_end": [],
        "suggested": 0,
        "released": [],
        "sync": [],
        "step_summaries": 0,
    }
    graph_obj = graph if graph is not None else FakeGraph(values)
    conv_obj = conv if conv is not None else make_conv()

    def _db():
        return NullDBSession()

    class _Repo:
        async def get_by_id_and_user(self, db, conv_id, user_id):
            return conv_obj

    class _Summary:
        async def maybe_compress(self, db, conv, model_id):
            return None

    async def _build_context(self, db, conv, model_id):
        recorder["ctx_calls"] += 1
        mems = injected_list if injected_list is not None else [{"memory_id": 99, "source": "preference"}]
        return [], "", mems

    async def _load_agent_anchor(self, db, conv):
        return "agent_a", "1"

    async def _load_snapshot(self, db, redis, agent_id, version_no):
        return snapshot if snapshot is not None else {"reasoning_mode": "react", "config": {}}

    async def _build_graph(self, db, redis, agent_id, version_no, model_id=None):
        return graph_obj

    async def _resolve(snapshot, messages, model_id):
        return resolve_mode

    async def _get_redis():
        return await fake_redis()

    async def _finalize(self, msg_id, result, model_id, used_memory_ids=None):
        recorder["finalized"].append((msg_id, result, model_id, used_memory_ids))
        return 0

    async def _push_end(self, stream_session_id, result, credits=0):
        recorder["pushed_end"].append(credits)

    def _suggest(self, conv_id, msg_id, result, user_id, stream_session_id):
        recorder["suggested"] += 1

    def _step_summaries(msg_id, model_id):
        recorder["step_summaries"] += 1

    def _sync(conv_id):
        recorder["sync"].append(conv_id)

    class _Emitter:
        async def release_lock(self, conv_id):
            recorder["released"].append(conv_id)

    monkeypatch.setattr("app.service.ai.reasoning_service.get_db_session", _db)
    monkeypatch.setattr("app.service.ai.reasoning_service.ai_conversation_repository", _Repo())
    monkeypatch.setattr("app.service.ai.reasoning_service.summary_service", _Summary())
    monkeypatch.setattr("app.service.ai.reasoning_service.context_manager", type("_Context", (), {"build_context": _build_context})())
    monkeypatch.setattr(
        "app.dependencies.redis.get_redis_client", _get_redis
    )
    if interrupt_tid is not None:

        class _AssertTidInterrupt:
            def __init__(self):
                self.saved = []
                self.cleared = []

            async def get_interrupt(self, thread_id):
                assert thread_id == interrupt_tid
                return interrupt

            async def save_interrupt(self, thread_id, itype, data):
                self.saved.append((thread_id, itype, data))

            async def clear_interrupt(self, thread_id):
                self.cleared.append(thread_id)

        handler = _AssertTidInterrupt()
    else:
        handler = StubInterruptHandler(interrupt)
    monkeypatch.setattr("app.service.ai.reasoning_service.interrupt_handler", handler)
    monkeypatch.setattr("app.service.ai.reasoning_service.sse_emitter_manager", _Emitter())
    monkeypatch.setattr(ReasoningService, "_load_agent_anchor", _load_agent_anchor)
    monkeypatch.setattr(ReasoningService, "_load_snapshot", _load_snapshot)
    monkeypatch.setattr(ReasoningService, "_build_graph", _build_graph)
    monkeypatch.setattr(DeepAgentBuilder, "resolve_reasoning_mode", _resolve)
    monkeypatch.setattr(ReasoningService, "_finalize_message", _finalize)
    monkeypatch.setattr(ReasoningService, "_push_end", _push_end)
    monkeypatch.setattr(ReasoningService, "_trigger_suggestions", _suggest)
    monkeypatch.setattr("app.service.ai.reasoning_service.schedule_step_summaries", _step_summaries)
    monkeypatch.setattr("app.service.ai.reasoning_service._schedule_conversation_sync", _sync)
    return ReasoningService(), recorder
