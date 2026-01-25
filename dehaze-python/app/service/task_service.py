"""
任务服务模块
实现导出任务的异步执行和管理
"""

import json
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from typing import List, Optional

from flask import current_app

from app.extensions import mysql
from app.models import (
    ExportTaskCreateForm,
    SysDataset,
    SysDatasetItem,
    SysFile,
    SysItemFile,
    SysTask,
    TaskStatus,
    TaskType,
    TaskVO,
)
from app.utils.error import BusinessException


class TaskService:
    """任务服务类"""

    # 缓存键前缀
    TASK_CACHE_PREFIX = "export:task:"
    TASK_CANCEL_PREFIX = "task:cancel:"
    TASK_EXPIRE_HOURS = 24  # 任务文件缓存24小时

    @staticmethod
    def create_export_task(form: ExportTaskCreateForm, user_id: int) -> TaskVO:
        """
        创建导出任务

        Args:
            form: 导出任务创建表单
            user_id: 当前用户ID

        Returns:
            任务VO

        Raises:
            BusinessException: 用户未登录或导出类型无效
        """
        if user_id is None:
            raise BusinessException("用户未登录")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务实体
        sys_task = SysTask(
            task_id=task_id,
            task_type=form.type,
            status=TaskStatus.PENDING,
            progress=0,
            total_files=0,
            processed_files=0,
            params=json.dumps({
                'type': form.type,
                'targetId': form.target_id,
                'targetIds': form.target_ids,
                'options': form.options
            }),
            created_by=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=TaskService.TASK_EXPIRE_HOURS)
        )

        # 保存任务到MySQL
        mysql.session.add(sys_task)
        mysql.session.flush()

        # 缓存任务信息到Redis
        cache_key = TaskService.TASK_CACHE_PREFIX + task_id
        task_dict = TaskService._task_to_dict(sys_task)
        redis_client = current_app.extensions['redis_client']
        redis_client.setex(
            cache_key,
            TaskService.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        # 转换为VO
        task_vo = TaskVO(sys_task)

        # 异步提交任务执行
        ThreadedTaskExecutor.submit_export_task(sys_task.id, form)

        current_app.logger.info(f"创建导出任务成功: taskId={task_id}, type={form.type}, userId={user_id}")

        return task_vo

    @staticmethod
    def get_task_status(task_id: str) -> Optional[TaskVO]:
        """
        查询任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务VO，如果任务不存在则返回None
        """
        if not task_id:
            raise BusinessException("任务ID不能为空")

        redis_client = current_app.extensions['redis_client']
        cache_key = TaskService.TASK_CACHE_PREFIX + task_id

        # 先从Redis缓存查询
        cached_task = redis_client.get(cache_key)
        if cached_task:
            try:
                task_data = json.loads(cached_task)
                return TaskVO._from_dict(task_data)
            except (json.JSONDecodeError, Exception) as e:
                current_app.logger.warning(f"解析缓存数据失败: {e}")

        # 从MySQL数据库查询
        sys_task = SysTask.query.filter_by(task_id=task_id).first()
        if sys_task is None:
            current_app.logger.warning(f"任务不存在: taskId={task_id}")
            return None

        # 更新缓存
        task_dict = TaskService._task_to_dict(sys_task)
        redis_client.setex(
            cache_key,
            TaskService.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        current_app.logger.info(f"查询任务状态: taskId={task_id}, status={sys_task.status}")

        return TaskVO(sys_task)

    @staticmethod
    def download_export_file(task_id: str) -> Optional[str]:
        """
        下载导出文件

        Args:
            task_id: 任务ID

        Returns:
            下载链接，如果任务未完成或已过期则返回None
        """
        if not task_id:
            raise BusinessException("任务ID不能为空")

        sys_task = TaskService._get_task_entity(task_id)
        if sys_task is None:
            return None

        # 检查任务状态
        if sys_task.status != TaskStatus.COMPLETED:
            current_app.logger.warning(f"任务未完成，无法下载: taskId={task_id}, status={sys_task.status}")
            return None

        # 检查任务是否过期
        if sys_task.expires_at and sys_task.expires_at < datetime.now():
            current_app.logger.warning(f"任务已过期，无法下载: taskId={task_id}, expiresAt={sys_task.expires_at}")
            return None

        # 从result字段获取下载链接
        if not sys_task.result:
            current_app.logger.warning(f"任务结果为空: taskId={task_id}")
            return None

        download_url = sys_task.result
        current_app.logger.info(f"生成下载链接: taskId={task_id}, url={download_url}")

        return download_url

    @staticmethod
    def cancel_task(task_id: str) -> None:
        """
        取消导出任务

        Args:
            task_id: 任务ID
        """
        if not task_id:
            raise BusinessException("任务ID不能为空")

        sys_task = TaskService._get_task_entity(task_id)
        if sys_task is None:
            raise BusinessException("任务不存在")

        # 检查任务状态
        if sys_task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            current_app.logger.warning(f"任务已完成或失败，无法取消: taskId={task_id}, status={sys_task.status}")
            return

        if sys_task.status == TaskStatus.CANCELLED:
            current_app.logger.warning(f"任务已取消: taskId={task_id}")
            return

        # 更新任务状态
        sys_task.status = TaskStatus.CANCELLED
        sys_task.completed_at = datetime.now()
        mysql.session.commit()

        # 更新缓存
        cache_key = TaskService.TASK_CACHE_PREFIX + task_id
        redis_client = current_app.extensions['redis_client']
        task_dict = TaskService._task_to_dict(sys_task)
        redis_client.setex(
            cache_key,
            TaskService.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

        # 设置取消标志位（通知执行器停止）
        cancel_key = TaskService.TASK_CANCEL_PREFIX + task_id
        redis_client.setex(cancel_key, 300, 'true')  # 5分钟过期

        current_app.logger.info(f"取消导出任务成功: taskId={task_id}")

    @staticmethod
    def _get_task_entity(task_id: str) -> Optional[SysTask]:
        """
        获取任务实体（优先从缓存获取）

        Args:
            task_id: 任务ID

        Returns:
            任务实体，如果不存在则返回None
        """
        redis_client = current_app.extensions['redis_client']
        cache_key = TaskService.TASK_CACHE_PREFIX + task_id

        cached_task = redis_client.get(cache_key)
        if cached_task:
            try:
                task_data = json.loads(cached_task)
                # 转换回实体
                sys_task = SysTask(
                    id=task_data.get('id'),
                    task_id=task_data.get('task_id'),
                    task_type=task_data.get('task_type'),
                    status=task_data.get('status'),
                    progress=task_data.get('progress'),
                    total_files=task_data.get('total_files'),
                    processed_files=task_data.get('processed_files'),
                    result=task_data.get('result'),
                    error_message=task_data.get('error_message'),
                    created_by=task_data.get('created_by'),
                    created_at=datetime.fromisoformat(task_data['created_at']) if task_data.get('created_at') else None,
                    started_at=datetime.fromisoformat(task_data['started_at']) if task_data.get('started_at') else None,
                    completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data.get(
                        'completed_at') else None,
                    expires_at=datetime.fromisoformat(task_data['expires_at']) if task_data.get('expires_at') else None,
                )
                return sys_task
            except (json.JSONDecodeError, Exception):
                pass

        return SysTask.query.filter_by(task_id=task_id).first()

    @staticmethod
    def _task_to_dict(task: SysTask) -> dict:
        """将任务实体转换为字典"""
        return {
            'id': task.id,
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status,
            'progress': task.progress,
            'total_files': task.total_files,
            'processed_files': task.processed_files,
            'result': task.result,
            'error_message': task.error_message,
            'created_by': task.created_by,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'expires_at': task.expires_at
        }


class ThreadedTaskExecutor:
    """线程化任务执行器 - 使用后台线程异步执行导出任务"""

    # 存储活跃任务 {task_id: thread}
    _active_tasks = {}
    _lock = threading.Lock()
    # 存储app上下文，供后台线程使用
    _app_context = None
    _app = None

    @staticmethod
    def set_app_context(app):
        """设置Flask应用上下文供后台线程使用"""
        ThreadedTaskExecutor._app = app

    @staticmethod
    def submit_export_task(db_task_id: int, form: ExportTaskCreateForm) -> None:
        """
        提交导出任务到后台线程

        Args:
            db_task_id: 数据库任务ID
            form: 导出任务创建表单
        """
        thread = threading.Thread(
            target=ThreadedTaskExecutor._execute_export_task,
            args=(db_task_id, form, ThreadedTaskExecutor._app),
            daemon=True
        )
        thread.start()

        # 注册活跃任务
        with ThreadedTaskExecutor._lock:
            sys_task = SysTask.query.get(db_task_id)
            if sys_task:
                ThreadedTaskExecutor._active_tasks[sys_task.task_id] = thread

    @staticmethod
    def _execute_export_task(db_task_id: int, form: ExportTaskCreateForm, app) -> None:
        """
        执行导出任务（在后台线程中运行）

        Args:
            db_task_id: 数据库任务ID
            form: 导出任务创建表单
            app: Flask应用实例
        """
        import logging
        logger = logging.getLogger(__name__)

        with app.app_context():
            logger.info(
                f"开始执行导出任务: taskId={db_task_id}, type={form.type}, "
                f"thread={threading.current_thread().name}"
            )

            # 查询任务
            sys_task = SysTask.query.get(db_task_id)
            if sys_task is None:
                logger.error(f"任务不存在: taskId={db_task_id}")
                return

            try:
                # 更新任务状态为processing
                sys_task.status = TaskStatus.PROCESSING
                sys_task.started_at = datetime.now()
                mysql.session.commit()

                # 更新缓存
                ThreadedTaskExecutor._update_cache(sys_task)

                # 根据导出类型执行不同的逻辑
                result = None
                if form.type == 'dataset':
                    result = ThreadedTaskExecutor._export_dataset(sys_task, form, app)
                elif form.type == 'dataset_item':
                    result = ThreadedTaskExecutor._export_dataset_item(sys_task, form, app)
                elif form.type == 'batch_items':
                    result = ThreadedTaskExecutor._export_batch_items(sys_task, form, app)
                elif form.type == 'custom':
                    result = ThreadedTaskExecutor._export_custom(sys_task, form, app)
                else:
                    logger.error(f"不支持的导出类型: type={form.type}")
                    ThreadedTaskExecutor._update_task_status(
                        db_task_id, TaskStatus.FAILED, None, "不支持的导出类型: " + form.type
                    )
                    return

                if result:
                    # 更新任务状态为completed
                    ThreadedTaskExecutor._update_task_status(db_task_id, TaskStatus.COMPLETED, result, None)
                    sys_task.status = TaskStatus.COMPLETED
                    sys_task.progress = 100
                    sys_task.result = result
                    sys_task.completed_at = datetime.now()
                    mysql.session.commit()
                    logger.info(f"导出任务完成: taskId={sys_task.task_id}, downloadUrl={result}")
                else:
                    ThreadedTaskExecutor._update_task_status(db_task_id, TaskStatus.FAILED, None, "导出失败")
                    sys_task.status = TaskStatus.FAILED
                    sys_task.completed_at = datetime.now()
                    mysql.session.commit()

            except Exception as e:
                if str(e) == "任务已被取消":
                    logger.warning(f"导出任务被取消: taskId={db_task_id}")
                    ThreadedTaskExecutor._update_task_status(db_task_id, TaskStatus.CANCELLED, None, None)
                    sys_task.status = TaskStatus.CANCELLED
                    sys_task.completed_at = datetime.now()
                else:
                    logger.error(f"导出任务执行失败: taskId={db_task_id}", exc_info=e)
                    ThreadedTaskExecutor._update_task_status(db_task_id, TaskStatus.FAILED, None, str(e))
                    sys_task.status = TaskStatus.FAILED
                    sys_task.error_message = str(e)
                    sys_task.completed_at = datetime.now()
                mysql.session.commit()

            finally:
                # 移除活跃任务
                task_id = sys_task.task_id  # 提前获取task_id，避免访问过期对象
                with ThreadedTaskExecutor._lock:
                    if task_id in ThreadedTaskExecutor._active_tasks:
                        del ThreadedTaskExecutor._active_tasks[task_id]

    @staticmethod
    def _export_dataset(sys_task: SysTask, form: ExportTaskCreateForm, app) -> Optional[str]:
        """导出单个数据集"""
        import logging
        logger = logging.getLogger(__name__)

        dataset_id = form.target_id
        if dataset_id is None:
            raise BusinessException("数据集ID不能为空")

        dataset = SysDataset.query.get(dataset_id)
        if dataset is None:
            raise BusinessException("数据集不存在")

        # 查询数据集下的所有数据项
        items = SysDatasetItem.query.filter_by(dataset_id=dataset_id).all()
        if not items:
            raise BusinessException("数据集为空")

        item_ids = [item.id for item in items]
        return ThreadedTaskExecutor._export_items_to_zip(
            sys_task, item_ids, f"{dataset.name}_export", form, app
        )

    @staticmethod
    def _export_dataset_item(sys_task: SysTask, form: ExportTaskCreateForm, app) -> Optional[str]:
        """导出单个数据项"""
        item_id = form.target_id
        if item_id is None:
            raise BusinessException("数据项ID不能为空")

        item = SysDatasetItem.query.get(item_id)
        if item is None:
            raise BusinessException("数据项不存在")

        return ThreadedTaskExecutor._export_items_to_zip(
            sys_task, [item_id], f"{item.name}_export", form, app
        )

    @staticmethod
    def _export_batch_items(sys_task: SysTask, form: ExportTaskCreateForm, app) -> Optional[str]:
        """批量导出数据项"""
        item_ids = form.target_ids
        if not item_ids:
            raise BusinessException("数据项ID列表不能为空")

        return ThreadedTaskExecutor._export_items_to_zip(
            sys_task, item_ids, f"batch_export_{uuid.uuid4().hex[:8]}", form, app
        )

    @staticmethod
    def _export_custom(sys_task: SysTask, form: ExportTaskCreateForm, app) -> Optional[str]:
        """自定义导出"""
        item_ids = form.target_ids
        if not item_ids:
            raise BusinessException("数据项ID列表不能为空")

        return ThreadedTaskExecutor._export_items_to_zip(
            sys_task, item_ids, f"custom_export_{uuid.uuid4().hex[:8]}", form, app
        )

    @staticmethod
    def _export_items_to_zip(
            sys_task: SysTask,
            item_ids: List[int],
            zip_name: str,
            form: ExportTaskCreateForm,
            app
    ) -> Optional[str]:
        """将数据项导出为ZIP文件"""
        import logging
        logger = logging.getLogger(__name__)

        # 检查取消标志位
        if ThreadedTaskExecutor._is_task_cancelled(sys_task.task_id, app):
            raise Exception("任务已被取消")

        # 创建临时目录
        temp_dir = os.path.join(app.config.get('TEMP_DIR', tempfile.gettempdir()), 'export')
        os.makedirs(temp_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, f"export_{zip_name}_{uuid.uuid4().hex[:8]}.zip")

        # MinIO对象名称（使用任务ID确保唯一性）
        object_name = f"exports/{sys_task.task_id}/{zip_name}.zip"

        # 获取导出选项
        options = form.options or {}
        structure = options.get('structure', 'by_item')
        include_types = options.get('includeTypes')
        include_thumbnail = options.get('includeThumbnail', False)

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zos:
                total_files = 0
                processed_files = 0

                # 预计算文件总数
                for item_id in item_ids:
                    item_files = SysItemFile.query.filter_by(item_id=item_id).all()
                    total_files += len(item_files)
                    if include_thumbnail:
                        total_files += len(item_files)

                sys_task.total_files = total_files
                mysql.session.commit()

                # 遍历每个数据项
                for item_id in item_ids:
                    # 检查取消标志位
                    if ThreadedTaskExecutor._is_task_cancelled(sys_task.task_id, app):
                        raise Exception("任务已被取消")

                    item = SysDatasetItem.query.get(item_id)
                    if item is None:
                        logger.warning(f"数据项不存在: itemId={item_id}")
                        continue

                    item_files = SysItemFile.query.filter_by(item_id=item_id).all()

                    # 根据组织结构添加文件
                    for item_file in item_files:
                        if ThreadedTaskExecutor._should_include_type(include_types, item_file.type):
                            ThreadedTaskExecutor._add_file_to_zip(
                                zos, item_file, structure, item.name, None, False, logger
                            )
                            processed_files += 1
                            ThreadedTaskExecutor._update_task_progress(
                                sys_task.id, processed_files, total_files
                            )

                        if include_thumbnail:
                            ThreadedTaskExecutor._add_file_to_zip(
                                zos, item_file, structure, item.name, "thumbnail", True, logger
                            )
                            processed_files += 1
                            ThreadedTaskExecutor._update_task_progress(
                                sys_task.id, processed_files, total_files
                            )

            logger.info(f"ZIP文件创建成功: file={zip_path}, totalFiles={processed_files}")

            # 上传文件到MinIO
            minio_client = app.extensions.get('minio_client')
            bucket_name = app.config.get('MINIO_BUCKET_NAME')

            if minio_client and bucket_name:
                # 上传ZIP文件
                with open(zip_path, 'rb') as f:
                    minio_client.put_object(
                        bucket_name,
                        object_name,
                        f,
                        length=os.path.getsize(zip_path),
                        content_type='application/zip'
                    )

                # 生成预签名URL（24小时有效）
                from datetime import timedelta
                download_url = minio_client.presigned_get_object(
                    bucket_name,
                    object_name,
                    expires=timedelta(hours=24)
                )

                logger.info(f"文件上传成功: objectName={object_name}, downloadUrl={download_url}")
                return download_url
            else:
                logger.warning("MinIO客户端未配置，返回空URL")
                return ""

        except Exception as e:
            logger.error(f"导出文件失败", exc_info=e)
            raise BusinessException("导出文件失败", e)
        finally:
            # 清理临时文件
            if os.path.exists(zip_path):
                os.remove(zip_path)

    @staticmethod
    def _should_include_type(include_types: Optional[List[str]], type: str) -> bool:
        """判断是否应该包含该类型"""
        if not include_types:
            return True
        return type in include_types

    @staticmethod
    def _add_file_to_zip(
            zos: zipfile.ZipFile,
            item_file: SysItemFile,
            structure: str,
            item_name: str,
            subfolder: Optional[str],
            is_thumbnail: bool,
            logger
    ) -> None:
        """
        添加文件到ZIP

        注意：这是一个简化实现，实际需要从文件存储系统下载文件内容
        """
        # 获取文件信息
        file_obj = SysFile.query.get(item_file.file_id)
        if file_obj is None:
            logger.warning(f"文件不存在: fileId={item_file.file_id}")
            return

        # 构建ZIP条目路径
        if structure == 'by_item':
            if subfolder:
                entry_path = f"{item_name}/{subfolder}/{item_file.id}.jpg"
            else:
                entry_path = f"{item_name}/{item_file.id}.jpg"
        else:
            if subfolder:
                entry_path = f"{subfolder}/{item_file.id}.jpg"
            else:
                entry_path = f"{item_file.id}.jpg"

        # 添加到ZIP（空文件，实际应该写入文件内容）
        zos.writestr(entry_path, '')

    @staticmethod
    def _update_task_status(
            db_task_id: int,
            status: str,
            result: Optional[str],
            error_message: Optional[str]
    ) -> None:
        """更新任务状态"""
        sys_task = SysTask.query.get(db_task_id)
        if sys_task:
            sys_task.status = status
            if result:
                sys_task.result = result
            if error_message:
                sys_task.error_message = error_message
            mysql.session.commit()

        # 更新缓存
        if sys_task:
            ThreadedTaskExecutor._update_cache(sys_task)

    @staticmethod
    def _update_task_progress(db_task_id: int, processed_files: int, total_files: int) -> None:
        """更新任务进度"""
        sys_task = SysTask.query.get(db_task_id)
        if sys_task:
            progress = int((processed_files * 100 / total_files)) if total_files > 0 else 100
            sys_task.progress = progress
            sys_task.processed_files = processed_files
            mysql.session.commit()

        # 更新缓存
        if sys_task:
            ThreadedTaskExecutor._update_cache(sys_task)

    @staticmethod
    def _update_cache(sys_task: SysTask) -> None:
        """更新缓存"""
        cache_key = TaskService.TASK_CACHE_PREFIX + sys_task.task_id
        redis_client = current_app.extensions['redis_client']
        task_dict = TaskService._task_to_dict(sys_task)
        redis_client.setex(
            cache_key,
            TaskService.TASK_EXPIRE_HOURS * 3600,
            json.dumps(task_dict, default=str)
        )

    @staticmethod
    def _is_task_cancelled(task_id: str, app=None) -> bool:
        """检查任务是否已被取消"""
        from app.extensions import redis_client

        cancel_key = TaskService.TASK_CANCEL_PREFIX + task_id
        is_cancelled = redis_client.get(cancel_key)
        # Redis返回的是bytes类型，需要解码
        if isinstance(is_cancelled, bytes):
            is_cancelled = is_cancelled.decode('utf-8')
        return is_cancelled == 'true'


# 临时文件导入
import tempfile
