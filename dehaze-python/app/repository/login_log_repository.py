from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.dependencies import mongo
from app.models.entity.mongo_log import LoginLogDocument
from app.repository.mongo_bson import as_int64, int64_candidates


class LoginLogRepository:
    """登录日志 Repository（MongoDB 实现）"""

    async def create_log(
        self,
        user_id: int | None,
        username: str,
        ip: str,
        status: int,
        message: str,
        browser: str = "",
        os: str = "",
        location: str = "",
        device_type: str = "web",
    ) -> dict:
        """写入一条登录日志。

        `user_id` 统一编码为 BSON int64：java（`Long`）与 go（`int64`）按该字段过滤登录
        日志（普通用户仅看本人），pymongo 对 < 2^31 的裸 int 写 int32 ⇒ 单类型读不到
        python 的记录且不报错（见 `app/repository/mongo_bson.py`）。
        `status` 保持 int32（java `Integer`、go `int` 驱动编码同为 int32，三端口径本就一致）。
        """
        doc = {
            "user_id": as_int64(user_id),
            "username": username,
            "ip": ip,
            "location": location,
            "browser": browser,
            "os": os,
            "device_type": device_type,
            "status": status,
            "message": message,
            "create_time": datetime.now(UTC),
        }
        result = await mongo.get_mongo_client()[settings.MONGODB_DATABASE][
            LoginLogDocument.COLLECTION
        ].insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    async def page_logs(
        self,
        page_num: int = 1,
        page_size: int = 10,
        *,
        username: str | None = None,
        ip: str | None = None,
        status: int | None = None,
        device_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_ids: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询登录日志（支持多条件筛选）

        - username/ip/device_type 精确匹配
        - status 精确匹配
        - start_time/end_time 限定 create_time 范围
        - user_ids 限定用户范围（普通用户仅查询本人日志时使用）
        """
        collection = mongo.get_mongo_client()[settings.MONGODB_DATABASE][
            LoginLogDocument.COLLECTION
        ]
        query = self._build_query(
            username=username,
            ip=ip,
            status=status,
            device_type=device_type,
            start_time=start_time,
            end_time=end_time,
            user_ids=user_ids,
        )
        total = await collection.count_documents(query)
        cursor = (
            collection.find(query)
            .sort("create_time", -1)
            .skip((page_num - 1) * page_size)
            .limit(page_size)
        )
        docs = [doc async for doc in cursor]
        return docs, total

    async def list_logs(
        self,
        *,
        username: str | None = None,
        ip: str | None = None,
        status: int | None = None,
        device_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """全量查询登录日志（导出用，按 create_time 倒序，条件同 page_logs）"""
        collection = mongo.get_mongo_client()[settings.MONGODB_DATABASE][
            LoginLogDocument.COLLECTION
        ]
        query = self._build_query(
            username=username,
            ip=ip,
            status=status,
            device_type=device_type,
            start_time=start_time,
            end_time=end_time,
            user_ids=user_ids,
        )
        cursor = collection.find(query).sort("create_time", -1)
        return [doc async for doc in cursor]

    @staticmethod
    def _build_query(
        *,
        username: str | None,
        ip: str | None,
        status: int | None,
        device_type: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
        user_ids: list[int] | None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if username:
            query["username"] = username
        if ip:
            query["ip"] = ip
        if status is not None:
            query["status"] = status
        if device_type:
            query["device_type"] = device_type
        if user_ids is not None:
            # int32/int64 双类型候选：本端新写入与另两端写入是 int64，历史 int32 记录也不能漏
            query["user_id"] = {"$in": int64_candidates(user_ids)}
        if start_time or end_time:
            time_range: dict[str, Any] = {}
            if start_time:
                time_range["$gte"] = start_time
            if end_time:
                time_range["$lte"] = end_time
            query["create_time"] = time_range
        return query


login_log_repository = LoginLogRepository()
