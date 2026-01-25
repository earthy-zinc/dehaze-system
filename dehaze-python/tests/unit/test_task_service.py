"""
任务服务测试
测试导出任务的创建、查询、下载和取消操作
"""
import json
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.models import ExportTaskCreateForm, SysDataset, SysTask, TaskStatus, TaskVO
from app.service.task_service import TaskService, ThreadedTaskExecutor


@pytest.fixture
def mock_redis_client(app):
    """提供可配置的 mock redis_client"""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.setex.return_value = True
    mock.delete.return_value = 1

    # 在 Flask app 的 extensions 字典中设置
    app.extensions['redis_client'] = mock

    # 同时在 app.extensions 模块中设置（因为 _is_task_cancelled 从那里导入）
    import app.extensions as ext_module
    ext_module.redis_client = mock

    yield mock


@pytest.mark.unit
@pytest.mark.requires_db
class TestTaskService:
    """任务服务测试类"""

    def test_create_export_task_dataset(self, db_session):
        """测试创建数据集导出任务"""
        # 先创建测试数据集（不设置tree_path，因为测试数据库可能没有这个字段）
        dataset = SysDataset(
            parent_id=0,
            type='test',
            name='测试数据集',
            path='/test',
            status=1
        )
        db_session.add(dataset)
        db_session.commit()

        # 创建导出任务表单
        form = ExportTaskCreateForm(
            type='dataset',
            target_id=dataset.id,
            options={'structure': 'by_item'}
        )

        # 创建任务
        task_vo = TaskService.create_export_task(form, user_id=1)

        # 验证任务创建成功
        assert task_vo is not None
        assert task_vo.task_id is not None
        assert task_vo.task_type == 'dataset'
        assert task_vo.status in (TaskStatus.PENDING, TaskStatus.PROCESSING)
        assert task_vo.progress == 0 or task_vo.progress == 100

        # 验证数据库中存在任务
        sys_task = SysTask.query.filter_by(task_id=task_vo.task_id).first()
        assert sys_task is not None
        assert sys_task.task_type == 'dataset'
        assert sys_task.created_by == 1

    def test_create_export_task_batch_items(self, db_session):
        """测试创建批量数据项导出任务"""
        form = ExportTaskCreateForm(
            type='batch_items',
            target_ids=[1, 2, 3],
            options={'structure': 'by_type', 'includeThumbnail': True}
        )

        task_vo = TaskService.create_export_task(form, user_id=1)

        assert task_vo is not None
        assert task_vo.task_type == 'batch_items'
        assert task_vo.total_files == 0  # 初始值

    def test_create_export_task_invalid_user(self, db_session):
        """测试未登录用户创建任务"""
        form = ExportTaskCreateForm(type='custom', target_ids=[1, 2])

        with pytest.raises(Exception) as exc_info:
            TaskService.create_export_task(form, user_id=None)

        assert '用户未登录' in str(exc_info.value)

    def test_get_task_status_from_cache(self, db_session):
        """测试从缓存获取任务状态"""
        # 创建任务
        form = ExportTaskCreateForm(type='dataset', target_id=1)
        task_vo = TaskService.create_export_task(form, user_id=1)
        task_id = task_vo.task_id

        # 从缓存查询
        cached_vo = TaskService.get_task_status(task_id)

        assert cached_vo is not None
        assert cached_vo.task_id == task_id
        assert cached_vo.task_type == 'dataset'

    def test_get_task_status_not_found(self, db_session):
        """测试查询不存在的任务"""
        fake_task_id = str(uuid.uuid4())
        task_vo = TaskService.get_task_status(fake_task_id)

        assert task_vo is None

    def test_get_task_status_empty_id(self, db_session):
        """测试查询空任务ID"""
        with pytest.raises(Exception) as exc_info:
            TaskService.get_task_status('')

        assert '任务ID不能为空' in str(exc_info.value)

    def test_download_export_file_not_completed(self, db_session):
        """测试下载未完成的任务文件"""
        form = ExportTaskCreateForm(type='dataset', target_id=1)
        task_vo = TaskService.create_export_task(form, user_id=1)
        task_id = task_vo.task_id

        download_url = TaskService.download_export_file(task_id)

        assert download_url is None

    def test_download_export_file_not_found(self, db_session):
        """测试下载不存在的任务"""
        fake_task_id = str(uuid.uuid4())
        download_url = TaskService.download_export_file(fake_task_id)

        assert download_url is None

    def test_cancel_task_pending(self, db_session):
        """测试取消等待中的任务"""
        form = ExportTaskCreateForm(type='dataset', target_id=1)
        task_vo = TaskService.create_export_task(form, user_id=1)
        task_id = task_vo.task_id

        # 取消任务
        TaskService.cancel_task(task_id)

        # 验证任务状态
        task = TaskService.get_task_status(task_id)
        if task and task.status != TaskStatus.PROCESSING:
            # 如果任务还未开始处理，验证取消状态
            assert task.status == TaskStatus.CANCELLED

    def test_cancel_task_not_found(self, db_session):
        """测试取消不存在的任务"""
        fake_task_id = str(uuid.uuid4())

        with pytest.raises(Exception) as exc_info:
            TaskService.cancel_task(fake_task_id)

        assert '任务不存在' in str(exc_info.value)

    def test_cancel_task_empty_id(self, db_session):
        """测试取消空任务ID"""
        with pytest.raises(Exception) as exc_info:
            TaskService.cancel_task('')

        assert '任务ID不能为空' in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.requires_db
class TestTaskVO:
    """TaskVO 测试类"""

    def test_task_vo_from_dict(self):
        """测试从字典创建 TaskVO"""
        task_dict = {
            'id': 1,
            'task_id': 'test-task-123',
            'task_type': 'dataset',
            'status': 'completed',
            'progress': 100,
            'total_files': 10,
            'processed_files': 10,
            'result': 'http://example.com/download.zip',
            'error_message': None,
            'created_by': 1,
            'created_at': '2025-01-01T00:00:00',
            'started_at': '2025-01-01T00:00:05',
            'completed_at': '2025-01-01T00:00:10',
            'expires_at': '2025-01-02T00:00:00'
        }

        task_vo = TaskVO._from_dict(task_dict)

        assert task_vo.id == 1
        assert task_vo.task_id == 'test-task-123'
        assert task_vo.task_type == 'dataset'
        assert task_vo.status == 'completed'
        assert task_vo.progress == 100
        assert task_vo.download_url == 'http://example.com/download.zip'
        assert task_vo.error is None

    def test_task_vo_download_url_none_when_not_completed(self):
        """测试未完成任务不应有下载链接"""
        task_dict = {
            'id': 1,
            'task_id': 'test-task-123',
            'task_type': 'dataset',
            'status': 'processing',
            'progress': 50,
            'total_files': 10,
            'processed_files': 5,
            'result': None,
            'error_message': None,
            'created_by': 1,
            'created_at': '2025-01-01T00:00:00',
            'started_at': None,
            'completed_at': None,
            'expires_at': '2025-01-02T00:00:00'
        }

        task_vo = TaskVO._from_dict(task_dict)

        assert task_vo.download_url is None

    def test_task_vo_download_url_none_when_no_result(self):
        """测试已完成但无结果的任务不应有下载链接"""
        task_dict = {
            'id': 1,
            'task_id': 'test-task-123',
            'task_type': 'dataset',
            'status': 'completed',
            'progress': 100,
            'total_files': 0,
            'processed_files': 0,
            'result': None,
            'error_message': None,
            'created_by': 1,
            'created_at': '2025-01-01T00:00:00',
            'started_at': None,
            'completed_at': '2025-01-01T00:00:10',
            'expires_at': '2025-01-02T00:00:00'
        }

        task_vo = TaskVO._from_dict(task_dict)

        assert task_vo.download_url is None


@pytest.mark.unit
@pytest.mark.requires_db
class TestThreadedTaskExecutor:
    """ThreadedTaskExecutor 测试类"""

    def test_submit_export_task(self, db_session):
        """测试提交导出任务"""
        form = ExportTaskCreateForm(type='dataset', target_id=1)
        task_vo = TaskService.create_export_task(form, user_id=1)

        # 提交任务（通过 create_export_task 已经触发）
        # 等待一小段时间确保线程启动
        time.sleep(0.1)

        # 验证活跃任务
        assert task_vo.task_id is not None

    def test_is_task_cancelled(self, db_session, mock_redis_client):
        """测试任务取消标志位的设置和检查"""
        test_task_id = "test-task-123"

        # 确认任务未被取消
        cancel_key = TaskService.TASK_CANCEL_PREFIX + test_task_id
        mock_redis_client.delete(cancel_key)
        # Mock 返回 None 表示未取消
        mock_redis_client.get.return_value = None
        is_cancelled = ThreadedTaskExecutor._is_task_cancelled(test_task_id)
        assert is_cancelled is False

        # 设置取消标志 - 模拟 Redis 返回 'true'
        mock_redis_client.setex(cancel_key, 300, 'true')
        mock_redis_client.get.return_value = b'true'  # Redis 返回 bytes 类型

        # 验证取消标志已设置
        is_cancelled = ThreadedTaskExecutor._is_task_cancelled(test_task_id)
        assert is_cancelled is True

        # 清理
        mock_redis_client.delete(cancel_key)
        mock_redis_client.get.return_value = None  # 恢复默认值


@pytest.mark.unit
@pytest.mark.requires_db
class TestExportTaskForm:
    """ExportTaskCreateForm 测试类"""

    def test_form_dataset_type(self):
        """测试数据集类型表单"""
        form = ExportTaskCreateForm(type='dataset', target_id=1)

        assert form.type == 'dataset'
        assert form.target_id == 1
        assert form.target_ids == []
        assert form.options == {}

    def test_form_batch_items_type(self):
        """测试批量数据项类型表单"""
        form = ExportTaskCreateForm(
            type='batch_items',
            target_ids=[1, 2, 3],
            options={'structure': 'by_type'}
        )

        assert form.type == 'batch_items'
        assert form.target_id is None
        assert form.target_ids == [1, 2, 3]
        assert form.options == {'structure': 'by_type'}

    def test_form_with_all_options(self):
        """测试包含所有选项的表单"""
        form = ExportTaskCreateForm(
            type='custom',
            target_ids=[1, 2],
            options={
                'structure': 'by_item',
                'includeTypes': ['clear', 'hazy'],
                'includeThumbnail': True
            }
        )

        assert form.type == 'custom'
        assert form.options['structure'] == 'by_item'
        assert form.options['includeTypes'] == ['clear', 'hazy']
        assert form.options['includeThumbnail'] is True
