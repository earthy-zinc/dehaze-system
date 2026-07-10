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
             patch("asyncio.create_task") as mock_create_task, \
             patch("app.service.task_tracker.get_task_tracker") as mock_get_tracker:
            mock_db.flush = AsyncMock()
            mock_db.refresh = AsyncMock()
            mock_db.add = MagicMock()  # add 是同步方法，不需要 AsyncMock

            # mock asyncio.create_task 返回一个可用的 task
            mock_task = AsyncMock()
            mock_task.add_done_callback = MagicMock()
            mock_create_task.return_value = mock_task

            # mock task tracker
            mock_tracker = AsyncMock()
            mock_tracker.register = AsyncMock()
            mock_get_tracker.return_value = mock_tracker

            result = await TaskServiceAsync.create_export_task(
                db=mock_db,
                redis=mock_redis,
                task_type="dataset",
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
        mock_redis.get = AsyncMock(return_value=b'{"task_id": "test-123", "status": "pending"}')

        result = await TaskServiceAsync.get_task_status(
            db=mock_db,
            redis=mock_redis,
            task_id="test-123",
        )

        assert result is not None
        assert result["task_id"] == "test-123"

    @pytest.mark.asyncio
    async def test_get_task_status_from_db(self):
        """测试从数据库获取任务状态"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_task = MagicMock()
            mock_task.task_id = "test-123"
            mock_task.status = "pending"
            mock_task.progress = 0
            mock_repo.get_by_task_id = AsyncMock(return_value=mock_task)

            result = await TaskServiceAsync.get_task_status(
                db=mock_db,
                redis=mock_redis,
                task_id="test-123",
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
            )

    @pytest.mark.asyncio
    async def test_cancel_task_success(self):
        """测试取消任务成功"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_task = MagicMock()
            mock_task.task_id = "test-123"
            mock_task.status = "pending"  # 使用字符串
            mock_repo.get_by_task_id = AsyncMock(return_value=mock_task)

            result = await TaskServiceAsync.cancel_task(
                db=mock_db,
                redis=mock_redis,
                task_id="test-123",
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
                )

    @pytest.mark.asyncio
    async def test_download_export_file_not_found(self):
        """测试下载导出文件时任务不存在"""
        mock_db = AsyncMock()
        mock_redis = AsyncMock()

        with patch("app.service.task_service.task_repository") as mock_repo:
            mock_repo.get_by_task_id = AsyncMock(return_value=None)

            result = await TaskServiceAsync.download_export_file(
                db=mock_db,
                redis=mock_redis,
                task_id="nonexistent",
            )

            assert result is None

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
                self.status = "pending"
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
        assert result["status"] == "pending"
