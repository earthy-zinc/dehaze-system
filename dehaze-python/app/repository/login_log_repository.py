from datetime import datetime, timezone

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
            "create_time": datetime.now(timezone.utc),
        }
        result = await get_mongo_client()[settings.MONGO_DB_NAME][LoginLogDocument.COLLECTION].insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc


login_log_repository = LoginLogRepository()
