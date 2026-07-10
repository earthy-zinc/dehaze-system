"""
自定义导出策略（桩实现）

预留扩展点，当前行为与批量下载一致。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.models.entity.sys_task import SysTask
from app.service.task.strategy import CancelChecker, ProgressCallback, TaskStrategy
from app.service.task.zip_utils import export_items_to_zip

logger = logging.getLogger(__name__)


class CustomExportStrategy(TaskStrategy):
    """自定义导出策略（桩）"""

    async def execute(
        self,
        db: AsyncSession,
        sys_task: SysTask,
        target_id: Optional[int],
        target_ids: Optional[List[int]],
        options: Dict[str, Any],
        progress_callback: ProgressCallback,
        cancel_checker: CancelChecker,
    ) -> Optional[str]:
        if not target_ids:
            raise BusinessException("数据项ID列表不能为空")

        return await export_items_to_zip(
            db, sys_task, target_ids,
            f"custom_export_{uuid.uuid4().hex[:8]}", options,
            progress_callback, cancel_checker,
        )
