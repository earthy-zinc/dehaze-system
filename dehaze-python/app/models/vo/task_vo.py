"""
任务视图对象
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.entity.sys_task import SysTask


class TaskVO:
    """任务视图对象"""

    def __init__(self, task: 'SysTask'):
        self.id = task.id
        self.task_id = task.task_id
        self.task_type = task.task_type
        self.status = task.status
        self.progress = task.progress
        self.total_files = task.total_files
        self.processed_files = task.processed_files
        # 只有completed状态且有结果时才有下载链接
        if task.status == 'completed' and task.result:
            self.download_url = task.result
        else:
            self.download_url = None
        self.error = task.error_message
        self.created_at = task.created_at.isoformat() if task.created_at else None
        self.started_at = task.started_at.isoformat() if task.started_at else None
        self.completed_at = task.completed_at.isoformat() if task.completed_at else None
        self.expires_at = task.expires_at.isoformat() if task.expires_at else None

    @classmethod
    def _from_dict(cls, task_dict: dict) -> 'TaskVO':
        """从字典创建TaskVO（用于缓存反序列化）"""
        task_vo = cls.__new__(cls)
        task_vo.id = task_dict.get('id')
        task_vo.task_id = task_dict.get('task_id')
        task_vo.task_type = task_dict.get('task_type')
        task_vo.status = task_dict.get('status')
        task_vo.progress = task_dict.get('progress', 0)
        task_vo.total_files = task_dict.get('total_files', 0)
        task_vo.processed_files = task_dict.get('processed_files', 0)
        # 只有completed状态且有结果时才有下载链接
        if task_dict.get('status') == 'completed' and task_dict.get('result'):
            task_vo.download_url = task_dict.get('result')
        else:
            task_vo.download_url = None
        task_vo.error = task_dict.get('error_message')
        task_vo.created_at = task_dict.get('created_at')
        task_vo.started_at = task_dict.get('started_at')
        task_vo.completed_at = task_dict.get('completed_at')
        task_vo.expires_at = task_dict.get('expires_at')
        return task_vo

    def to_dict(self):
        return {
            'id': self.id,
            'taskId': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'totalFiles': self.total_files,
            'processedFiles': self.processed_files,
            'downloadUrl': self.download_url,
            'error': self.error,
            'createdAt': self.created_at,
            'startedAt': self.started_at,
            'completedAt': self.completed_at,
            'expiresAt': self.expires_at
        }
