from typing import Optional

from fastapi import Depends, HTTPException, status
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
    nickname: Optional[str] = None
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


def _extract_permissions(payload: dict) -> list[str]:
    """
    从 JWT payload 提取权限列表

    Args:
        payload: JWT payload

    Returns:
        权限列表
    """
    permissions = payload.get("permissions") or payload.get("perms") or payload.get("perms_list", [])

    if isinstance(permissions, str):
        return [p.strip() for p in permissions.split(",") if p.strip()]
    elif isinstance(permissions, list):
        return [str(p) for p in permissions if p]
    return []


def _extract_roles(payload: dict) -> list[str]:
    """
    从 JWT payload 的 authorities 字段提取角色列表

    Args:
        payload: JWT payload

    Returns:
        角色列表（去掉 ROLE_ 前缀）
    """
    authorities = payload.get("authorities")
    if not authorities:
        return []

    if isinstance(authorities, list):
        return [str(a).replace("ROLE_", "", 1) if str(a).startswith("ROLE_") else str(a) for a in authorities]
    elif isinstance(authorities, str):
        return [a.strip().replace("ROLE_", "", 1) if a.strip().startswith("ROLE_") else a.strip() for a in authorities.split(",") if a.strip()]
    return []


async def get_current_user(
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
    username = payload.get("sub") or payload.get("username", "")

    user_context = UserContext(
        id=int(user_id),
        username=username,
        nickname=payload.get("nickname"),
        roles=_extract_roles(payload),
        permissions=_extract_permissions(payload),
    )

    set_current_user_id(user_context.id)

    return user_context


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
) -> Optional[UserContext]:
    """
    获取当前登录用户（可选）

    用于可选认证场景，如公开接口但登录用户有额外权限

    Args:
        credentials: Bearer Token

    Returns:
        用户上下文，未登录返回 None
    """
    if credentials is None:
        return None

    try:
        payload = decode_token(credentials.credentials)

        jti = payload.get("jti")
        if jti:
            from app.dependencies.redis import get_redis_client
            redis = await get_redis_client()
            if redis:
                is_blacklisted = await redis.get(f"token:blacklist:{jti}")
                if is_blacklisted:
                    return None

        user_id = payload.get("userId")
        if not user_id:
            return None

        # sub 统一为 username（与 Java/Go 一致）
        username = payload.get("sub") or payload.get("username", "")

        return UserContext(
            id=int(user_id),
            username=username,
            nickname=payload.get("nickname"),
            roles=_extract_roles(payload),
            permissions=_extract_permissions(payload),
        )
    except HTTPException:
        return None
