"""AI 兼容 API 调用审计日志 Repository（MongoDB `ai_api_call_log`）

只追加不更新，TTL 30 天自动过期。查询服务于登录用户查自己（user_id 强制过滤）。
"""

import asyncio
import logging
from datetime import UTC, datetime

from app.config import settings
from app.dependencies import mongo
from app.models.entity.mongo_log import AiApiCallLogDocument

logger = logging.getLogger(__name__)


class MongoAiCallLogRepository:
    """AI 兼容 API 调用审计 Repository（MongoDB 实现）"""

    @property
    def _collection(self):
        return mongo.get_mongo_client()[settings.MONGODB_DATABASE][AiApiCallLogDocument.COLLECTION]

    async def insert(self, doc: dict) -> str:
        """插入一条调用审计记录，返回 _id 字符串"""
        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    def insert_async(self, **fields) -> None:
        """异步写入调用审计（不阻塞业务主流程，失败时记录 warn 日志）。

        字段缺省值在此补全；user_id/key_id 允许为 None（401 被拒调用无凭证上下文）。
        """

        async def _write():
            try:
                doc = {
                    "user_id": fields.get("user_id"),
                    "key_id": fields.get("key_id"),
                    "key_prefix": fields.get("key_prefix", ""),
                    "conversation_id": fields.get("conversation_id"),
                    "model": fields.get("model"),
                    "endpoint": fields.get("endpoint", ""),
                    "protocol": fields.get("protocol", ""),
                    "is_stream": bool(fields.get("is_stream", False)),
                    "input_tokens": fields.get("input_tokens") or 0,
                    "output_tokens": fields.get("output_tokens") or 0,
                    "credits": fields.get("credits"),
                    "status_code": fields.get("status_code", 500),
                    "duration_ms": fields.get("duration_ms") or 0,
                    "client_ip": fields.get("client_ip", ""),
                    "request_id": fields.get("request_id", ""),
                    "error_msg": fields.get("error_msg"),
                    "create_time": datetime.now(UTC),
                }
                await self.insert(doc)
            except Exception as e:  # noqa: BLE001 - 审计失败不影响业务主流程
                logger.warning(
                    "调用审计写入失败 endpoint=%s status_code=%s: %s",
                    fields.get("endpoint"),
                    fields.get("status_code"),
                    e,
                )

        asyncio.create_task(_write())

    async def query(
        self,
        user_id: int,
        key_id: int | None = None,
        model: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict], int]:
        """按用户/Key/模型/时间筛选调用日志（create_time 倒序分页）。

        服务对象为登录用户查自己，user_id 强制过滤。
        """
        filter_cond: dict = {"user_id": user_id}
        if key_id is not None:
            filter_cond["key_id"] = key_id
        if model:
            filter_cond["model"] = model
        time_cond: dict = {}
        if start_time is not None:
            time_cond["$gte"] = start_time
        if end_time is not None:
            time_cond["$lte"] = end_time
        if time_cond:
            filter_cond["create_time"] = time_cond

        cursor = (
            self._collection.find(filter_cond)
            .sort("create_time", -1)
            .skip((page - 1) * size)
            .limit(size)
        )
        records = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            records.append(doc)
        total = await self._collection.count_documents(filter_cond)
        return records, total

    async def stats_by_key(
        self, user_id: int, start_time: datetime, end_time: datetime
    ) -> list[dict]:
        """按 key_id 聚合调用统计（总调用/成功数/失败数/总 tokens/总 credits），供对账汇总。"""
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "create_time": {"$gte": start_time, "$lte": end_time},
                }
            },
            {
                "$group": {
                    "_id": "$key_id",
                    "total_calls": {"$sum": 1},
                    "success_calls": {"$sum": {"$cond": [{"$eq": ["$status_code", 200]}, 1, 0]}},
                    "failed_calls": {"$sum": {"$cond": [{"$ne": ["$status_code", 200]}, 1, 0]}},
                    "total_tokens": {"$sum": {"$add": ["$input_tokens", "$output_tokens"]}},
                    "total_credits": {"$sum": {"$ifNull": ["$credits", 0]}},
                }
            },
        ]
        result = []
        async for doc in self._collection.aggregate(pipeline):
            result.append(
                {
                    "key_id": doc["_id"],
                    "total_calls": doc["total_calls"],
                    "success_calls": doc["success_calls"],
                    "failed_calls": doc["failed_calls"],
                    "total_tokens": doc["total_tokens"],
                    "total_credits": doc["total_credits"],
                }
            )
        return result


mongo_ai_call_log_repository = MongoAiCallLogRepository()
