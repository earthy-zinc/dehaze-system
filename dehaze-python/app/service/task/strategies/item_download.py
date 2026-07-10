"""
单个数据项下载策略

导出单个数据项的文件为 ZIP。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessException
from app.models.entity.sys_task import SysTask
from app.repository.dataset_repository import dataset_repository
from app.service.task.strategy import CancelChecker, ProgressCallback, TaskStrategy
from app.service.task.zip_utils import export_items_to_zip

logger = logging.getLogger(__name__)


class ItemDownloadStrategy(TaskStrategy):
    """单个数据项下载策略"""

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
        if target_id is None:
            raise BusinessException("数据项ID不能为空")

        item = await dataset_repository.get_item_by_id(db, target_id)
        if item is None:
            raise BusinessException("数据项不存在")

        return await export_items_to_zip(
            db, sys_task, [target_id], f"{item.name}_export", options,
            progress_callback, cancel_checker,
        )
