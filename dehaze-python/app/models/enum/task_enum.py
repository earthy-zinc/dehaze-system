"""
任务相关枚举
"""

from enum import Enum


class TaskStatus(str, Enum):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    CANCELLED = 'CANCELLED'


class TaskType(str, Enum):
    DATASET_EXPORT = 'dataset_export'
    USER_EXPORT = 'user_export'
    ROLE_EXPORT = 'role_export'
    DEPT_EXPORT = 'dept_export'
    MENU_EXPORT = 'menu_export'
    DICT_EXPORT = 'dict_export'
    ALGORITHM_EXPORT = 'algorithm_export'

    USER_IMPORT = 'user_import'
    ROLE_IMPORT = 'role_import'
    DEPT_IMPORT = 'dept_import'
    MENU_IMPORT = 'menu_import'
    DICT_IMPORT = 'dict_import'
    ALGORITHM_IMPORT = 'algorithm_import'


EXPORT_TASK_TYPES = {
    TaskType.DATASET_EXPORT.value,
    TaskType.USER_EXPORT.value,
    TaskType.ROLE_EXPORT.value,
    TaskType.DEPT_EXPORT.value,
    TaskType.MENU_EXPORT.value,
    TaskType.DICT_EXPORT.value,
    TaskType.ALGORITHM_EXPORT.value,
}

IMPORT_TASK_TYPES = {
    TaskType.USER_IMPORT.value,
    TaskType.ROLE_IMPORT.value,
    TaskType.DEPT_IMPORT.value,
    TaskType.MENU_IMPORT.value,
    TaskType.DICT_IMPORT.value,
    TaskType.ALGORITHM_IMPORT.value,
}


def get_task_category(task_type: str) -> str | None:
    if task_type in EXPORT_TASK_TYPES:
        return 'export'
    if task_type in IMPORT_TASK_TYPES:
        return 'import'
    return None


def get_module_by_type(task_type: str) -> str | None:
    if task_type.endswith('_import'):
        return task_type[:-7]
    if task_type.endswith('_export'):
        return task_type[:-7]
    return None
