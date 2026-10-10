import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.dependencies import mongo
from app.models.entity.mongo_log import AuditLogDocument
from app.repository.mongo_bson import as_int64, int64_or_int32_in

logger = logging.getLogger(__name__)


def _target_id_filter(target_type: str, target_id: Any) -> dict[str, Any]:
    """构造按目标对象匹配的过滤条件，整型 target_id 用 int32/int64 双类型 $in。

    java（Long）与 go（int64）写入/查询的 target_id 是 int64，而 pymongo 会把小于 2^31 的
    裸 int 编码为 int32——BSON 数值类型不相等，沿用裸 `{"target_id": target_id}` 读不到
    另两端写入的记录且**不报错**。故两种编码都列为候选（与 go
    `audit_log_repository.ListByTarget` 同法）。非整型目标（订单号/充值单号字符串、
    批量 id 列表）按单类型精确匹配。
    """
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        return {"target_type": target_type, "target_id": target_id}
    return {
        "target_type": target_type,
        "target_id": int64_or_int32_in(target_id),
    }


class MongoAuditLogRepository:
    """业务操作审计日志 Repository（MongoDB 实现，白名单驱动）"""

    @property
    def _collection(self):
        return mongo.get_mongo_client()[settings.MONGODB_DATABASE][AuditLogDocument.COLLECTION]

    async def create_audit(
        self,
        operator_id: int,
        target_type: str,
        target_id: Any,
        action: str,
        module: str,
        before_value: Any = None,
        after_value: Any = None,
        ip: str = "",
        user_agent: str = "",
    ) -> dict:
        """写入一条审计日志。

        `operator_id` / `target_id`（整型时）统一编码为 BSON int64：java 用 `Long`、
        go 用 `int64` 查这两个字段，而 pymongo 对 < 2^31 的裸 int 写 int32 ⇒ 读不到
        python 的记录且不报错（见 `app/repository/mongo_bson.py`）。
        """
        doc = {
            "operator_id": as_int64(operator_id),
            "target_type": target_type,
            "target_id": as_int64(target_id),
            "action": action,
            "module": module,
            "before_value": before_value,
            "after_value": after_value,
            "ip": ip,
            "user_agent": user_agent,
            "create_time": datetime.now(UTC),
        }
        result = await self._collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    async def list_by_target(
        self, target_type: str, target_id: Any, page: int, page_size: int
    ) -> tuple[list[dict], int]:
        """按目标对象分页查询审计日志（create_time 倒序）"""
        filt = _target_id_filter(target_type, target_id)
        total = await self._collection.count_documents(filt)
        cursor = (
            self._collection.find(filt)
            .sort("create_time", -1)
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        return [doc async for doc in cursor], total

    def create_audit_async(self, **kwargs) -> None:
        """异步写入审计日志（不阻塞业务主流程，失败时记录 warn 日志）。

        任务持有强引用（事件循环对 task 仅弱引用，不持有会被 GC 中途丢弃），
        完成后自动移出引用集。
        """
        task = asyncio.create_task(self._write_audit(**kwargs))
        _BACKGROUND_AUDIT_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_AUDIT_TASKS.discard)

    async def _write_audit(self, **kwargs) -> None:
        try:
            await self.create_audit(**kwargs)
        except Exception as e:
            logger.warning(
                "审计日志写入失败 module=%s action=%s: %s",
                kwargs.get("module"),
                kwargs.get("action"),
                e,
            )


_BACKGROUND_AUDIT_TASKS: set[asyncio.Task] = set()


mongo_audit_log_repository = MongoAuditLogRepository()
