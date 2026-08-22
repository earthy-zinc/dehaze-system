"""
API Key 认证中间件

职责：拦截请求中的 Authorization: Bearer dhak_* 凭证，校验后注入用户上下文。
与 Session 认证完全解耦，遵循单一职责原则。
"""

import hashlib
from datetime import datetime

from fastapi import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.core.code import ResultCode
from app.models.base import set_current_user_id

# 兼容 API 端点路径（OpenAI/Claude 双协议）：401 审计仅记录这些路径的拒绝，
# 避免非兼容端点的 M2M 调用拒绝污染 ai_api_call_log
_COMPAT_ENDPOINTS = frozenset(
    {
        "/api/v1/chat/completions",
        "/api/v1/models",
        "/api/v1/messages",
    }
)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件，独立于 Session 认证"""

    @staticmethod
    def _audit_401(request: Request, token: str, error_msg: str) -> None:
        """兼容端点 401 拒绝审计（key_id=None，prefix 取前 8 字符，不存完整 Key）。

        record_call 采用函数内局部导入：middleware 加载早于 service 层，
        模块级导入会在应用启动期反向拉起 service 依赖链。
        """
        path = request.url.path
        if path not in _COMPAT_ENDPOINTS:
            return
        from app.service.ai.compatible_audit import record_call

        record_call(
            user_id=None,
            key_id=None,
            key_prefix=token[:8],
            conversation_id=None,
            model=None,
            endpoint=path.rsplit("/", 1)[-1],
            # Anthropic 协议经 x-api-key 携带；Bearer/Authorization 路径按 OpenAI 记
            protocol="claude" if request.headers.get("x-api-key") else "openai",
            is_stream=False,
            status_code=401,
            error_msg=error_msg,
            client_ip=request.client.host if request.client else "",
        )

    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization", "")
        x_api_key = request.headers.get("x-api-key")

        # OpenAI 协议：Authorization: Bearer dhak_xxx
        token = None
        if auth_header.startswith("Bearer ") and auth_header[7:].startswith("dhak_"):
            token = auth_header[7:]
        # Anthropic 协议：x-api-key 直接携带原始 key（无 Bearer 前缀）
        elif x_api_key and (x_api_key.startswith("dhak_") or x_api_key.startswith("sk-ant")):
            token = x_api_key

        if not token:
            return await call_next(request)

        # 校验 API Key
        key_hash = hashlib.sha256(token.encode()).hexdigest()

        from sqlalchemy import select

        from app.models.entity.api_key import SysApiKey

        db = getattr(request.state, "db", None)
        if db is None:
            self._audit_401(request, token, "认证服务不可用")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "code": ResultCode.TOKEN_INVALID.code,
                    "msg": "认证服务不可用",
                    "data": None,
                },
            )

        # 仅匹配未吊销的 key（revoked_at IS NULL），吊销的 key 视同不存在
        stmt = select(SysApiKey).where(
            SysApiKey.key_hash == key_hash,
            SysApiKey.revoked_at.is_(None),
        )
        result = await db.execute(stmt)
        api_key = result.scalar_one_or_none()

        if not api_key or api_key.status != 1:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "code": ResultCode.TOKEN_INVALID.code,
                    "msg": ResultCode.TOKEN_INVALID.msg,
                    "data": None,
                },
            )
        if api_key.expires_at is not None and api_key.expires_at <= datetime.now():
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "code": ResultCode.TOKEN_INVALID.code,
                    "msg": ResultCode.TOKEN_INVALID.msg,
                    "data": None,
                },
            )

        # 构建用户上下文
        from app.dependencies.redis import get_redis_client
        from app.repository.role_repository import role_repository
        from app.repository.user_repository import user_repository
        from app.service.menu_service import menu_service

        user = await user_repository.get_by_id(db, api_key.user_id)
        if not user or user.status != 1:
            self._audit_401(request, token, "API Key 所属用户无效或已禁用")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "code": ResultCode.ACCESS_UNAUTHORIZED.code,
                    "msg": ResultCode.ACCESS_UNAUTHORIZED.msg,
                    "data": None,
                },
            )

        roles = await user_repository.get_user_role_codes(db, user.id)
        redis = await get_redis_client()
        # Redis 不可用时降级为最小权限：data_scope=1（仅本人）、perms 为空集，
        # 与 perms 的降级方向保持一致，避免 Redis 故障期间获得最大数据权限
        data_scope = await role_repository.get_maximum_data_scope(db, roles) if redis else 1
        perms = await menu_service.list_role_perms(db, redis, set(roles)) if redis else set()

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
        # 注入 API Key 标识（供接入治理/调用审计使用）
        request.state.api_key_info = {
            "key_id": api_key.id,
            "key_prefix": api_key.key_prefix or "",
        }
        set_current_user_id(user.id)

        return await call_next(request)
