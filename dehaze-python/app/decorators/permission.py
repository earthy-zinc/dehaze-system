import fnmatch
from functools import wraps
from typing import Callable

from fastapi import HTTPException, status

from app.core.code import ResultCode
from app.dependencies.auth import UserContext


def _match_permission(user_permissions: list[str], required_permission: str) -> bool:
    """
    检查单个权限（支持通配符）

    Args:
        user_permissions: 用户权限列表
        required_permission: 需要的权限

    Returns:
        是否具有权限
    """
    # 超级管理员权限
    if "*" in user_permissions or "*:*" in user_permissions:
        return True

    if required_permission in user_permissions:
        return True

    for user_perm in user_permissions:
        if fnmatch.fnmatch(required_permission, user_perm):
            return True
        if fnmatch.fnmatch(user_perm, required_permission):
            return True

    return False


def require_permission(permission: str):
    """
    权限检查装饰器（单一权限）

    使用示例:
        @router.post("/", response_model=Result[dict])
        @require_permission("sys:user:add")
        async def create_user(...):
            ...

    Args:
        permission: 所需权限标识，如 "sys:user:add"

    Raises:
        HTTPException: 用户无权限时抛出 403
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: UserContext, **kwargs):
            if user.is_root:
                return await func(*args, user=user, **kwargs)

            if not _match_permission(user.permissions, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=ResultCode.FORBIDDEN_OPERATION.msg,
                )

            return await func(*args, user=user, **kwargs)

        return wrapper

    return decorator


def require_any_permission(*permissions: str):
    """
    权限检查装饰器（任意一个权限即可）

    使用示例:
        @router.get("/page")
        @require_any_permission("sys:user:view", "sys:user:list")
        async def get_user_page(...):
            ...

    Args:
        permissions: 所需权限标识列表，满足其中之一即可

    Raises:
        HTTPException: 用户无任何所需权限时抛出 403
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: UserContext, **kwargs):
            if user.is_root:
                return await func(*args, user=user, **kwargs)

            for perm in permissions:
                if _match_permission(user.permissions, perm):
                    return await func(*args, user=user, **kwargs)

            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=ResultCode.FORBIDDEN_OPERATION.msg,
            )

        return wrapper

    return decorator


def require_all_permissions(*permissions: str):
    """
    权限检查装饰器（需要所有权限）

    使用示例:
        @router.post("/admin/batch")
        @require_all_permissions("sys:user:add", "sys:user:edit")
        async def batch_update_users(...):
            ...

    Args:
        permissions: 所需权限标识列表，必须全部满足

    Raises:
        HTTPException: 用户缺少任一权限时抛出 403
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: UserContext, **kwargs):
            if user.is_root:
                return await func(*args, user=user, **kwargs)

            for perm in permissions:
                if not _match_permission(user.permissions, perm):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=ResultCode.FORBIDDEN_OPERATION.msg,
                    )

            return await func(*args, user=user, **kwargs)

        return wrapper

    return decorator
