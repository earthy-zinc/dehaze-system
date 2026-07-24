from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
from pydantic import BaseModel

from app.config import settings
from app.core.code import ResultCode
from app.models.base import set_current_user_id

oauth2_scheme = HTTPBearer(auto_error=False)


class UserContext(BaseModel):
    id: int
    username: str
    dept_id: Optional[int] = None
    data_scope: Optional[int] = None
    roles: list[str] = []
    permissions: list[str] = []

    @property
    def is_root(self) -> bool:
        return "ROOT" in self.roles

    @property
    def is_admin(self) -> bool:
        return "ROOT" in self.roles or "ADMIN" in self.roles


def decode_token(token: str) -> dict:
    """
    解码 JWT Token

    Args:
        token: JWT Token 字符串

    Returns:
        Token payload

    Raises:
        HTTPException: Token 无效或过期
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=["HS256"],
            options={"require_exp": True},  # 强制验证 exp
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )
    except JWTError as e:
        # JWTError 包含所有 JWT 相关错误（包括 claims 验证失败）
        error_msg = str(e) if str(e) else ResultCode.TOKEN_INVALID.msg
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg,
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )


def _split_authorities(payload: dict) -> tuple[list[str], list[str]]:
    """
    从 JWT payload 的 authorities 字段拆分出角色与权限

    元素带 ROLE_ 前缀的为角色，其余为权限（与 Go 一致：权限合并进 authorities）

    Args:
        payload: JWT payload

    Returns:
        (角色列表, 权限列表)
    """
    authorities = payload.get("authorities") or []
    if isinstance(authorities, str):
        authorities = [a.strip() for a in authorities.split(",") if a.strip()]

    roles = [str(a)[len("ROLE_"):] for a in authorities if str(a).startswith("ROLE_")]
    perms = [str(a) for a in authorities if not str(a).startswith("ROLE_")]
    return roles, perms


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
) -> UserContext:
    """
    获取当前登录用户（必需）

    Args:
        credentials: Bearer Token

    Returns:
        用户上下文

    Raises:
        HTTPException: 未登录或 Token 无效
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.ACCESS_UNAUTHORIZED.msg,
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    if token.startswith("dhak_"):
        from app.service.api_key_service import ApiKeyService
        user_context = await ApiKeyService.authenticate_by_key(
            request.state.db, token)
        set_current_user_id(user_context.id)
        return user_context

    # 先解码 Token 获取 jti
    payload = decode_token(token)

    jti = payload.get("jti")
    if jti:
        from app.dependencies.redis import get_redis_client
        redis = await get_redis_client()
        if redis:
            is_blacklisted = await redis.get(f"token:blacklist:{jti}")
            if is_blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token 已失效，请重新登录",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )

    user_id = payload.get("userId")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.TOKEN_INVALID.msg,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # sub 统一为 username（与 Java/Go 一致）
    username = payload.get("sub", "")

    roles, perms = _split_authorities(payload)
    user_context = UserContext(
        id=int(user_id),
        username=username,
        dept_id=payload.get("deptId"),
        data_scope=payload.get("dataScope"),
        roles=roles,
        permissions=perms,
    )

    set_current_user_id(user_context.id)

    return user_context


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
) -> Optional[UserContext]:
    if credentials is None:
        return None

    token = credentials.credentials

    if token.startswith("dhak_"):
        from app.service.api_key_service import ApiKeyService
        user_context = await ApiKeyService.authenticate_by_key(
            request.state.db, token)
        set_current_user_id(user_context.id)
        return user_context

    payload = decode_token(token)

    jti = payload.get("jti")
    if jti:
        from app.dependencies.redis import get_redis_client
        redis = await get_redis_client()
        if redis:
            is_blacklisted = await redis.get(f"token:blacklist:{jti}")
            if is_blacklisted:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token 已失效，请重新登录",
                    headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
                )

    user_id = payload.get("userId")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.TOKEN_INVALID.msg,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # sub 统一为 username（与 Java/Go 一致）
    username = payload.get("sub", "")

    roles, perms = _split_authorities(payload)
    user_context = UserContext(
        id=int(user_id),
        username=username,
        dept_id=payload.get("deptId"),
        data_scope=payload.get("dataScope"),
        roles=roles,
        permissions=perms,
    )

    set_current_user_id(user_context.id)

    return user_context
