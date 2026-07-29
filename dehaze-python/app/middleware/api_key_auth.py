"""
API Key 认证中间件

职责：拦截请求中的 Authorization: Bearer dhak_* 凭证，校验后注入用户上下文。
与 Session 认证完全解耦，遵循单一职责原则。
"""
import json
import hashlib
from datetime import datetime

from fastapi import status
from app.core.code import ResultCode
from app.models.base import set_current_user_id
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件，独立于 Session 认证"""

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        token = auth_header[7:]
        if not token.startswith("dhak_"):
            return await call_next(request)

        # 校验 API Key
        key_hash = hashlib.sha256(token.encode()).hexdigest()

        from app.models.entity.api_key import SysApiKey
        from sqlalchemy import select

        db = getattr(request.state, "db", None)
        if db is None:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"code": ResultCode.TOKEN_INVALID.code, "msg": "认证服务不可用", "data": None},
            )

        stmt = select(SysApiKey).where(SysApiKey.key_hash == key_hash)
        result = await db.execute(stmt)
        api_key = result.scalar_one_or_none()

        if not api_key or api_key.status != 1:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"code": ResultCode.TOKEN_INVALID.code, "msg": ResultCode.TOKEN_INVALID.msg, "data": None},
            )
        if api_key.expires_at is not None and api_key.expires_at <= datetime.now():
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"code": ResultCode.TOKEN_INVALID.code, "msg": ResultCode.TOKEN_INVALID.msg, "data": None},
            )

        # 构建用户上下文
        from app.repository.user_repository import user_repository
        from app.repository.role_repository import role_repository
        from app.service.menu_service import MenuService
        from app.dependencies.redis import get_redis_client

        user = await user_repository.get_by_id(db, api_key.user_id)
        if not user or user.status != 1:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"code": ResultCode.ACCESS_UNAUTHORIZED.code, "msg": ResultCode.ACCESS_UNAUTHORIZED.msg, "data": None},
            )

        roles = await user_repository.get_user_role_codes(db, user.id)
        redis = await get_redis_client()
        data_scope = await role_repository.get_maximum_data_scope(db, roles) if redis else 0
        perms = await MenuService.list_role_perms(db, redis, set(roles)) if redis else set()

        # 更新最后使用时间
        api_key.last_used_at = datetime.now()
        await db.flush()

        # 注入用户上下文到 request.state
        request.state.user_context = {
            "id": user.id,
            "username": user.username or "",
            "dept_id": user.dept_id,
            "data_scope": data_scope,
            "roles": roles,
            "permissions": list(perms),
            "is_m2m": True,
        }
        set_current_user_id(user.id)

        return await call_next(request)
