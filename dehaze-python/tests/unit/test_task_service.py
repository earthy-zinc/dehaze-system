"""
任务服务测试

测试 TaskServiceAsync 的核心功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.service.task_service import TaskServiceAsync
from app.core.exceptions import BusinessException


@pytest.mark.unit
class TestTaskService:
    """任务服务测试"""

    @pytest.mark.asyncio
    async def test_create_export_task(self):
        """测试创建导出任务"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo, \
             patch.object(TaskServiceAsync, "_dispatch_task", new_callable=AsyncMock) as mock_dispatch:
            mock_db.flush = AsyncMock()
            mock_db.refresh = AsyncMock()
            mock_db.add = MagicMock()  # add 是同步方法，不需要 AsyncMock

            result = await TaskServiceAsync.create_export_task(
                db=mock_db,
                redis=mock_redis,
                task_type="dataset_export",
                target_id=1,
                target_ids=None,
                options=None,
                user_id=1,
            )

            assert result is not None
            assert "task_id" in result

    @pytest.mark.asyncio
    async def test_create_export_task_no_user(self):
        """测试创建导出任务时用户未登录"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="用户未登录"):
            await TaskServiceAsync.create_export_task(
                db=mock_db,
                redis=mock_redis,
                task_type="dataset",
                target_id=1,
                target_ids=None,
                options=None,
                user_id=None,
            )

    @pytest.mark.asyncio
    async def test_get_task_status_from_cache(self):
        """测试从缓存获取任务状态"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=b'{"task_id": "test-123", "status": "PENDING", "created_by": 1}')

        result = await TaskServiceAsync.get_task_status(
            db=mock_db,
            redis=mock_redis,
            task_id="test-123",
            user_id=1,
        )

        assert result is not None
        assert result["task_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_get_task_status_from_db(self):
        """测试从数据库获取任务状态"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_task = MagicMock()
            mock_task.task_id = "test-123"
            mock_task.status = "PENDING"
            mock_task.progress = 0
            mock_task.created_by = 1
            mock_task.id = 1
            mock_task.task_type = "dataset_export"
            mock_task.total_files = 0
            mock_task.processed_files = 0
            mock_task.result = None
            mock_task.error_message = None
            mock_task.created_at = None
            mock_task.started_at = None
            mock_task.completed_at = None
            mock_task.expires_at = None
            mock_repo.get_by_task_id = AsyncMock(return_value=mock_task)

            result = await TaskServiceAsync.get_task_status(
                db=mock_db,
                redis=mock_redis,
                task_id="test-123",
                user_id=1,
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_get_task_status_empty_id(self):
        """测试获取任务状态时 ID 为空"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="任务ID不能为空"):
            await TaskServiceAsync.get_task_status(
                db=mock_db,
                redis=mock_redis,
                task_id="",
                user_id=1,
            )

    @pytest.mark.asyncio
    async def test_cancel_task_success(self):
        """测试取消任务成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_task = MagicMock()
            mock_task.task_id = "test-123"
            mock_task.status = "PENDING"  # 使用字符串
            mock_task.created_by = 1
            mock_task.id = 1
            mock_task.task_type = "dataset_export"
            mock_task.progress = 0
            mock_task.total_files = 0
            mock_task.processed_files = 0
            mock_task.result = None
            mock_task.error_message = None
            mock_task.created_at = None
            mock_task.started_at = None
            mock_task.completed_at = None
            mock_task.expires_at = None
            mock_repo.get_by_task_id = AsyncMock(return_value=mock_task)

            result = await TaskServiceAsync.cancel_task(
                db=mock_db,
                redis=mock_redis,
                task_id="test-123",
                user_id=1,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self):
        """测试取消任务时任务不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_repo.get_by_task_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="任务不存在"):
                await TaskServiceAsync.cancel_task(
                    db=mock_db,
                    redis=mock_redis,
                    task_id="nonexistent",
                    user_id=1,
                )

    @pytest.mark.asyncio
    async def test_download_export_file_not_found(self):
        """测试下载导出文件时任务不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_repo.get_by_task_id = AsyncMock(return_value=None)

            with pytest.raises(BusinessException, match="任务不存在"):
                await TaskServiceAsync.download_export_file(
                    db=mock_db,
                    redis=mock_redis,
                    task_id="nonexistent",
                    user_id=1,
                )

    @pytest.mark.asyncio
    async def test_download_export_file_empty_id(self):
        """测试下载导出文件时 ID 为空"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with pytest.raises(BusinessException, match="任务ID不能为空"):
            await TaskServiceAsync.download_export_file(
                db=mock_db,
                redis=mock_redis,
                task_id="",
                user_id=1,
            )


@pytest.mark.unit
class TestTaskUtils:
    """任务工具函数测试"""

    def test_task_to_dict(self):
        """测试任务实体转字典"""
        class MockTask:
            def __init__(self):
                self.id = 1
                self.task_id = "test-123"
                self.task_type = "dataset"
                self.status = "PENDING"
                self.progress = 0
                self.total_files = 10
                self.processed_files = 0
                self.result = None
                self.error_message = None
                self.created_by = 1
                self.created_at = None
                self.started_at = None
                self.completed_at = None
                self.expires_at = None

        mock_task = MockTask()
        result = TaskServiceAsync._task_to_dict(mock_task)

        assert result["task_id"] == "test-123"
        assert result["task_type"] == "dataset"
        assert result["status"] == "PENDING"
