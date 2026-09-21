"""Redis 检查点 Saver 测试：键 TTL、待写合并原子性、读取回环

aput_writes 的并发合并是重点：GET→SET 读改写下并发写同一 checkpoint 会互相
覆盖（实测 50 并发丢 49 条），LangGraph 恢复依赖完整的 pending writes 列表。
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import TASKS
from langgraph.types import Send
from redis.asyncio import Redis

from app.infrastructure.cache import checkpoint_manager as cm

_TTL = 30 * 24 * 3600


@pytest.fixture
def saver():
    return cm.RedisSaver(serde=JsonPlusSerializer())


def _checkpoint(cid: str = "ckpt-1") -> dict:
    return {
        "id": cid,
        "ts": "2026-09-17T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _config(tid: str = "t1", cid: str = "ckpt-1") -> dict:
    return {"configurable": {"thread_id": tid, "checkpoint_ns": "", "checkpoint_id": cid}}


class _YieldingRedis:
    """模拟真实 Redis 的网络往返：每条命令前让出事件循环。

    fakeredis 命令不产生 I/O，纯 CPU 协程下并发命令不会交错；真实 Redis 每条
    命令都是一次网络往返，GET→SET 读改写必然交错。
    """

    def __init__(self, inner: Redis):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        # 透明代理：动态转发任意 Redis 命令，返回类型随被代理命令而变。
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        # 替身：被代理的 fakeredis 命令均为协程，显式窄化为 Awaitable 以 await
        call = cast("Callable[..., Awaitable[Any]]", attr)

        async def _call(*args, **kwargs):
            await asyncio.sleep(0)
            return await call(*args, **kwargs)

        return _call


@pytest.fixture
def yielding_saver(mock_redis, monkeypatch):
    """走"有网络往返"的 Redis 的 Saver（用于并发竞态断言）"""
    redis = _YieldingRedis(mock_redis)

    async def _client():
        return redis

    monkeypatch.setattr(cm, "get_redis_client", _client)
    return cm.RedisSaver(serde=JsonPlusSerializer())


class TestKeyTtl:
    async def test_aput_sets_ttl_on_checkpoint_and_latest(self, mock_redis, saver):
        await saver.aput(_config(), _checkpoint(), {}, {})
        assert await mock_redis.ttl("ai:checkpoint:t1::ckpt-1") == _TTL
        assert await mock_redis.ttl("ai:checkpoint:latest:t1:") == _TTL

    async def test_aput_writes_sets_ttl(self, mock_redis, saver):
        await saver.aput_writes(_config(), [("messages", "hi")], "task-1")
        assert await mock_redis.ttl("ai:checkpoint:writes:t1::ckpt-1") == _TTL

    async def test_ttl_refreshed_on_rewrite(self, mock_redis, saver):
        await saver.aput(_config(), _checkpoint(), {}, {})
        await mock_redis.expire("ai:checkpoint:t1::ckpt-1", 10)
        await saver.aput(_config(), _checkpoint(), {}, {})
        assert await mock_redis.ttl("ai:checkpoint:t1::ckpt-1") == _TTL


class TestAputWritesAtomicity:
    @pytest.fixture(autouse=True)
    async def _checkpoint_exists(self, mock_redis, saver):
        await saver.aput(_config(), _checkpoint(), {}, {})

    async def test_concurrent_writes_all_persisted(self, yielding_saver):
        """并发写同一 checkpoint：合并须原子，读改写交错不得丢 writes"""
        await yielding_saver.aput(_config(), _checkpoint(), {}, {})
        await asyncio.gather(
            *[
                yielding_saver.aput_writes(_config(), [(f"ch-{i}", f"v{i}")], f"task-{i}")
                for i in range(50)
            ]
        )
        tuple_ = await yielding_saver.aget_tuple(_config())
        assert len(tuple_.pending_writes) == 50
        assert {w[0] for w in tuple_.pending_writes} == {f"task-{i}" for i in range(50)}

    async def test_concurrent_duplicate_task_ids_written_once(self, yielding_saver):
        """并发重复提交同一 task 的 writes：去重语义不得被并发破坏"""
        await yielding_saver.aput(_config(), _checkpoint(), {}, {})
        await asyncio.gather(
            *[
                yielding_saver.aput_writes(_config(), [("messages", f"v{i}")], "task-1")
                for i in range(20)
            ]
        )
        tuple_ = await yielding_saver.aget_tuple(_config())
        assert len(tuple_.pending_writes) == 1

    async def test_duplicate_task_write_upserts_new_value(self, mock_redis, saver):
        """同 (task_id, idx) 二次写按官方 upsert 语义覆盖，不得保留旧值"""
        await saver.aput_writes(_config(), [("messages", "a")], "task-1")
        await saver.aput_writes(_config(), [("messages", "b")], "task-1")
        tuple_ = await saver.aget_tuple(_config())
        assert [w[2] for w in tuple_.pending_writes] == ["b"]

    async def test_upsert_keeps_position_and_leaves_others(self, mock_redis, saver):
        """覆盖只替换命中的那一条，其余 writes 与顺序不受影响"""
        await saver.aput_writes(_config(), [("messages", "a"), ("tool", "keep")], "task-1")
        await saver.aput_writes(_config(), [("messages", "a2")], "task-1")
        tuple_ = await saver.aget_tuple(_config())
        assert [w[2] for w in tuple_.pending_writes] == ["a2", "keep"]

    async def test_distinct_write_idx_under_same_task_both_kept(self, mock_redis, saver):
        await saver.aput_writes(_config(), [("messages", "a"), ("tool", "b")], "task-1")
        tuple_ = await saver.aget_tuple(_config())
        assert [w[2] for w in tuple_.pending_writes] == ["a", "b"]

    async def test_empty_writes_leaves_key_unset(self, mock_redis, saver):
        await saver.aput_writes(_config(), [], "task-1")
        assert await mock_redis.get("ai:checkpoint:writes:t1::ckpt-1") is None
        assert (await saver.aget_tuple(_config())) is not None  # 待写为空不破坏检查点读取


class TestRoundTrip:
    async def test_aget_tuple_reads_checkpoint_and_writes(self, mock_redis, saver):
        await saver.aput(_config(), _checkpoint(), {}, {})
        await saver.aput_writes(_config(), [("messages", "hello")], "task-1", "path/0")
        tuple_ = await saver.aget_tuple(_config())
        assert tuple_.checkpoint["id"] == "ckpt-1"
        assert [(w[0], w[1], w[2]) for w in tuple_.pending_writes] == [
            ("task-1", "messages", "hello")
        ]

    async def test_send_write_roundtrip(self, mock_redis, saver):
        """Send 写入（__pregel_tasks 通道）无损回读。

        当前 langgraph（1.2.x）的 CheckpointTuple 没有 pending_sends 字段，官方
        InMemorySaver 也把 TASKS 通道写入一并返回在 pending_writes 中，运行时从
        pending_writes 消费 Send。pending_sends 仅存在于 checkpoint v<4 的旧格式
        （已迁移到 checkpoint["channel_values"][TASKS]）。若升级后协议新增该字段，
        本用例会失败并提示重新评估拆分逻辑。
        """
        await saver.aput(_config(), _checkpoint(), {}, {})
        await saver.aput_writes(_config(), [(TASKS, Send(node="worker", arg={"q": 1}))], "task-1")
        tuple_ = await saver.aget_tuple(_config())
        assert not hasattr(tuple_, "pending_sends")
        assert len(tuple_.pending_writes) == 1
        task_id, channel, value = tuple_.pending_writes[0]
        assert (task_id, channel) == ("task-1", TASKS)
        assert value == Send(node="worker", arg={"q": 1})

    async def test_aget_tuple_without_checkpoint_id_uses_latest(self, mock_redis, saver):
        await saver.aput(_config(cid="c1"), _checkpoint("c1"), {}, {})
        await saver.aput(_config(cid="c2"), _checkpoint("c2"), {}, {})
        tuple_ = await saver.aget_tuple({"configurable": {"thread_id": "t1"}})
        assert tuple_.checkpoint["id"] == "c2"
