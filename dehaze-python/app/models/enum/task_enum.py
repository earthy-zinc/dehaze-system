"""
任务相关枚举
"""

from enum import IntEnum, StrEnum


class TaskStatus(IntEnum):
    PENDING = 1
    PROCESSING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5


class TaskType(StrEnum):
    DATASET_EXPORT = "dataset_export"
    USER_EXPORT = "user_export"
    ROLE_EXPORT = "role_export"
    DEPT_EXPORT = "dept_export"
    MENU_EXPORT = "menu_export"
    DICT_EXPORT = "dict_export"
    ALGORITHM_EXPORT = "algorithm_export"

    USER_IMPORT = "user_import"
    ROLE_IMPORT = "role_import"
    DEPT_IMPORT = "dept_import"
    MENU_IMPORT = "menu_import"
    DICT_IMPORT = "dict_import"
    ALGORITHM_IMPORT = "algorithm_import"


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
