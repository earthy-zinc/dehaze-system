import fnmatch
from collections.abc import Callable
from functools import wraps

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
    if "*" in user_permissions or "*:*" in user_permissions:
        return True

    if required_permission in user_permissions:
        return True

    for user_perm in user_permissions:
        if fnmatch.fnmatchcase(required_permission, user_perm):
            return True
        if fnmatch.fnmatchcase(user_perm, required_permission):
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
