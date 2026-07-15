"""
装饰器模块

提供权限检查装饰器，统一安全校验机制
"""

from app.decorators.permission import require_permission

__all__ = [
    "require_permission",
]
