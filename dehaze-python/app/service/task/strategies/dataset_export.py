"""
数据集导出策略

导出单个数据集下的所有数据项为 ZIP 文件。
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


class DatasetExportStrategy(TaskStrategy):
    """数据集导出策略"""

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
            raise BusinessException("数据集ID不能为空")

        dataset = await dataset_repository.get_by_id(db, target_id)
        if dataset is None:
            raise BusinessException("数据集不存在")

        items = await dataset_repository.get_items_by_dataset_id(db, target_id)
        if not items:
            raise BusinessException("数据集为空")

        item_ids = [item.id for item in items]
        return await export_items_to_zip(
            db, sys_task, item_ids, f"{dataset.name}_export", options,
            progress_callback, cancel_checker,
        )
