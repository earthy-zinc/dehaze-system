"""
Repository 层

提供数据访问抽象层，每张表对应一个 Repository。
Service 层通过 Repository 访问数据，实现业务逻辑与数据访问的解耦。
"""

from app.repository.algorithm_repository import algorithm_repository
from app.repository.base import BaseRepository
from app.repository.dataset_repository import dataset_repository
from app.repository.dept_repository import dept_repository
from app.repository.dict_repository import dict_repository
from app.repository.file_repository import file_repository
from app.repository.login_log_repository import login_log_repository
from app.repository.menu_repository import menu_repository
from app.repository.mongo_audit_log_repository import mongo_audit_log_repository
from app.repository.role_repository import role_repository
from app.repository.task_repository import task_repository
from app.repository.user_repository import user_repository

__all__ = [
    "BaseRepository",
    "user_repository",
    "role_repository",
    "menu_repository",
    "dept_repository",
    "dict_repository",
    "algorithm_repository",
    "dataset_repository",
    "file_repository",
    "task_repository",
    "login_log_repository",
    "mongo_audit_log_repository",
]
