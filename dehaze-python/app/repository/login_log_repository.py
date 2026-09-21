from datetime import UTC, datetime
from typing import Any

from app.config import settings
from app.dependencies import mongo
from app.models.entity.mongo_log import LoginLogDocument


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
        doc = {
            "user_id": user_id,
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
            query["user_id"] = {"$in": user_ids}
        if start_time or end_time:
            time_range: dict[str, Any] = {}
            if start_time:
                time_range["$gte"] = start_time
            if end_time:
                time_range["$lte"] = end_time
            query["create_time"] = time_range
        return query


login_log_repository = LoginLogRepository()
