from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from app.config import settings
from app.dependencies.mongo import get_mongo_client
from app.models.entity.mongo_log import LoginLogDocument


class LoginLogRepository:
    """登录日志 Repository（MongoDB 实现）"""

    async def create_log(
        self,
        db,
        user_id: int | None,
        username: str,
        ip: str,
        status: int,
        message: str,
        browser: str = "",
        os: str = "",
        location: str = "",
    ) -> dict:
        doc = {
            "user_id": user_id,
            "username": username,
            "ip": ip,
            "location": location,
            "browser": browser,
            "os": os,
            "status": status,
            "message": message,
            "create_time": datetime.now(UTC),
        }
        result = await get_mongo_client()[settings.MONGO_DB_NAME][
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
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_ids: list[int] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页查询登录日志（支持多条件筛选）

        - username/ip 精确匹配
        - status 精确匹配
        - start_time/end_time 限定 create_time 范围
        - user_ids 限定用户范围（普通用户仅查询本人日志时使用）
        """
        collection = get_mongo_client()[settings.MONGO_DB_NAME][LoginLogDocument.COLLECTION]

        query: dict[str, Any] = {}
        if username:
            query["username"] = username
        if ip:
            query["ip"] = ip
        if status is not None:
            query["status"] = status
        if user_ids is not None:
            query["user_id"] = {"$in": user_ids}
        if start_time or end_time:
            time_range: dict[str, Any] = {}
            if start_time:
                time_range["$gte"] = start_time
            if end_time:
                time_range["$lte"] = end_time
            query["create_time"] = time_range

        total = await collection.count_documents(query)
        cursor = collection.find(query).sort("create_time", -1).skip((page_num - 1) * page_size).limit(page_size)
        docs = [doc async for doc in cursor]
        return docs, total


login_log_repository = LoginLogRepository()
