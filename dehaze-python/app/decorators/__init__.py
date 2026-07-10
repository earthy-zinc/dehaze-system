"""
装饰器模块

提供权限检查装饰器，统一权限校验机制
"""

from app.decorators.permission import (
    require_permission,
    require_any_permission,
    require_all_permissions,
)

__all__ = [
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
]
