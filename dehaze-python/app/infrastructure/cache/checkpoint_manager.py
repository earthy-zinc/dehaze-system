"""检查点管理器（CheckpointManager）

运行时存储：Redis（高速读写、interrupt/resume）。

- RedisSaver：自定义实现，继承 BaseCheckpointSaver，使用项目 get_redis_client()。
  由于当前 langgraph 版本（1.2.x）未内置 RedisSaver，此处按 LangGraph 标准
  Checkpointer 协议实现，支撑中断/恢复即可，不做完整时间旅行。
- CheckpointManager：提供运行时 checkpointer（Redis）。

注：Redis 客户端为异步，因此 Saver 实现异步方法（aget_tuple/aput/aput_writes），
同步方法保持基类默认（抛 NotImplementedError），推理统一走 ainvoke 异步路径。
"""

import base64
import json
import logging
from collections.abc import Awaitable, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from app.dependencies.redis import get_redis_client

logger = logging.getLogger(__name__)

# 检查点键 TTL（秒）。运行时键可重建，中断后未清理的线程（用户不再恢复）若
# 无 TTL 会永久占用内存；业务层无清理入口，TTL 是唯一回收手段
_CHECKPOINT_TTL = 30 * 24 * 3600

# 待写合并（Lua 原子执行）：GET→SET 读改写下并发写同一 checkpoint 会互相覆盖，
# 丢失的 pending writes 会导致 LangGraph 恢复时丢步骤。同 (task_id, idx) 的二次
# 写按官方 saver 的 upsert 语义覆盖旧值（重跑/修正写必须生效，保留旧值会回放旧数据）
_MERGE_WRITES_LUA = """
local raw = redis.call('GET', KEYS[1])
local existing = {}
if raw then existing = cjson.decode(raw) end
local seen = {}
for i = 1, #existing do
    seen[existing[i][1] .. ':' .. tostring(existing[i][2])] = i
end
local incoming = cjson.decode(ARGV[1])
for i = 1, #incoming do
    local w = incoming[i]
    local k = w[1] .. ':' .. tostring(w[2])
    local pos = seen[k]
    if pos then
        existing[pos] = w
    else
        existing[#existing + 1] = w
        seen[k] = #existing
    end
end
if #existing == 0 then
    return 0
end
redis.call('SET', KEYS[1], cjson.encode(existing), 'EX', ARGV[2])
return #existing
"""


def _configurable(config: RunnableConfig) -> dict[str, Any]:
    """读取 config 的 configurable 段。

    RunnableConfig 中该键为可选（total=False），直接下标访问会抛 KeyError；
    缺失时按空 dict 处理，与原「config 未带 configurable 即崩溃」相比改为显式空值，
    调用方随后对 thread_id 的取值同样会因缺失而暴露（取值语义不变）。
    """
    return config.get("configurable") or {}


class RedisSaver(BaseCheckpointSaver):
    """基于项目 Redis 的运行时检查点 Saver

    存储结构（均为字符串键值，30 天 TTL）：
    - 检查点:  ai:checkpoint:{thread_id}:{ns}:{checkpoint_id}  -> {type, blob, parent}
    - 最新指针: ai:checkpoint:latest:{thread_id}:{ns}          -> checkpoint_id
    - 待写:    ai:checkpoint:writes:{thread_id}:{ns}:{checkpoint_id} -> JSON 列表
    """

    _CHECKPOINT_KEY = "ai:checkpoint:{tid}:{ns}:{cid}"
    _LATEST_KEY = "ai:checkpoint:latest:{tid}:{ns}"
    _WRITES_KEY = "ai:checkpoint:writes:{tid}:{ns}:{cid}"

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        conf = _configurable(config)
        thread_id = conf["thread_id"]
        checkpoint_ns = conf.get("checkpoint_ns", "")
        checkpoint_id = c["id"]
        parent_id = conf.get("checkpoint_id")

        type_str, blob = self.serde.dumps_typed(c)
        redis = await get_redis_client()
        key = self._CHECKPOINT_KEY.format(tid=thread_id, ns=checkpoint_ns, cid=checkpoint_id)
        await redis.set(
            key,
            json.dumps(
                {
                    "type": type_str,
                    "blob": base64.b64encode(blob).decode("ascii"),
                    "parent": parent_id,
                    "metadata": json.dumps(get_checkpoint_metadata(config, metadata)),
                }
            ),
            ex=_CHECKPOINT_TTL,
        )
        await redis.set(
            self._LATEST_KEY.format(tid=thread_id, ns=checkpoint_ns),
            checkpoint_id,
            ex=_CHECKPOINT_TTL,
        )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        conf = _configurable(config)
        thread_id = conf["thread_id"]
        checkpoint_ns = conf.get("checkpoint_ns", "")
        checkpoint_id = conf["checkpoint_id"]
        redis = await get_redis_client()
        key = self._WRITES_KEY.format(tid=thread_id, ns=checkpoint_ns, cid=checkpoint_id)
        incoming = []
        for idx, (channel, value) in enumerate(writes):
            type_str, blob = self.serde.dumps_typed(value)
            incoming.append(
                [
                    task_id,
                    WRITES_IDX_MAP.get(channel, idx),
                    channel,
                    type_str,
                    base64.b64encode(blob).decode("ascii"),
                    task_path,
                ]
            )
        # redis-py 同步/异步客户端共用 eval 签名（标注 Union[Awaitable[str], str]）：
        # 此处为异步客户端，运行时恒返回协程，await 正确；类型系统无法区分同步/异步，
        # 用 cast 收窄到 Awaitable（非消音，await 语义与运行时一致）。
        await cast(
            Awaitable[str],
            redis.eval(_MERGE_WRITES_LUA, 1, key, json.dumps(incoming), str(_CHECKPOINT_TTL)),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        conf = _configurable(config)
        thread_id = conf["thread_id"]
        checkpoint_ns = conf.get("checkpoint_ns", "")
        redis = await get_redis_client()

        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            checkpoint_id = await redis.get(
                self._LATEST_KEY.format(tid=thread_id, ns=checkpoint_ns)
            )
        if not checkpoint_id:
            return None

        key = self._CHECKPOINT_KEY.format(tid=thread_id, ns=checkpoint_ns, cid=checkpoint_id)
        raw = await redis.get(key)
        if not raw:
            return None
        entry = json.loads(raw)
        checkpoint: Checkpoint = self.serde.loads_typed(
            (entry["type"], base64.b64decode(entry["blob"]))
        )

        writes_raw = await redis.get(
            self._WRITES_KEY.format(tid=thread_id, ns=checkpoint_ns, cid=checkpoint_id)
        )
        pending_writes = []
        if writes_raw:
            for task_id, _idx, channel, type_str, blob, _path in sorted(
                json.loads(writes_raw), key=lambda x: x[1]
            ):
                pending_writes.append(
                    (task_id, channel, self.serde.loads_typed((type_str, base64.b64decode(blob))))
                )

        parent_id = entry.get("parent")
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=json.loads(entry.get("metadata", "{}")),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=pending_writes,
        )

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.0000000000000000"


class CheckpointManager:
    """检查点管理器：提供运行时 Redis Saver"""

    def __init__(self) -> None:
        self._redis_saver = RedisSaver(serde=JsonPlusSerializer())

    def get_checkpointer(self) -> RedisSaver:
        """返回运行时 checkpointer（Redis），供 LangGraph 编译使用"""
        return self._redis_saver


# 全局单例
checkpoint_manager = CheckpointManager()
