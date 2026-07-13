"""
装饰器/依赖模块

提供权限检查、限流、防重复提交等装饰器/依赖，统一安全校验机制
"""

from app.decorators.permission import (
    require_permission,
    require_any_permission,
    require_all_permissions,
)
from app.decorators.rate_limit import rate_limit
from app.decorators.repeat_submit import repeat_submit

__all__ = [
    "require_permission",
    "require_any_permission",
    "require_all_permissions",
    "rate_limit",
    "repeat_submit",
]
