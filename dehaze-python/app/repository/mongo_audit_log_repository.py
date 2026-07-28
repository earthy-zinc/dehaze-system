import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.dependencies.mongo import get_mongo_client
from app.models.entity.mongo_log import AuditLogDocument

logger = logging.getLogger(__name__)


class MongoAuditLogRepository:
    """业务操作审计日志 Repository（MongoDB 实现，白名单驱动）"""

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
        doc = {
            "operator_id": operator_id,
            "target_type": target_type,
            "target_id": target_id,
            "action": action,
            "module": module,
            "before_value": before_value,
            "after_value": after_value,
            "ip": ip,
            "user_agent": user_agent,
            "create_time": datetime.now(timezone.utc),
        }
        result = await get_mongo_client()[settings.MONGO_DB_NAME][AuditLogDocument.COLLECTION].insert_one(doc)
        doc["_id"] = result.inserted_id
        return doc

    def create_audit_async(self, **kwargs) -> None:
        """异步写入审计日志（不阻塞业务主流程，失败时记录 warn 日志）"""
        async def _write():
            try:
                await self.create_audit(**kwargs)
            except Exception as e:
                logger.warning("审计日志写入失败 module=%s action=%s: %s", kwargs.get("module"), kwargs.get("action"), e)

        asyncio.create_task(_write())


mongo_audit_log_repository = MongoAuditLogRepository()
