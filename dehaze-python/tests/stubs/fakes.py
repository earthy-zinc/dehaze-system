"""仿真假件：实现真实接口协议的桩对象（DB/图/LLM/SSE/中断/仓储等）。

本模块只放"实现了真实协议"的假件类与对象（fakes），即对外暴露的属性方法
与真实实现对齐，可在测试中替代真实依赖。纯数据构造与 patch 辅助见
tests.stubs.factories / tests.stubs.mocks。
"""

from types import SimpleNamespace


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


class LLMChunk:
    def __init__(self, type="text_delta", content="", usage=None):
        self.type = type
        self.content = content
        self.usage = usage


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


class MemberBenefitRepo:
    def __init__(self, member=None, benefit=None):
        self.member = member
        self.benefit = benefit

    async def get_by_user_id(self, db, user_id):
        return self.member

    async def get_by_level_code(self, db, level_code):
        return self.benefit
