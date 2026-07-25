import json
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.core.code import ResultCode
from app.dependencies.redis import get_redis_client
from app.models.base import set_current_user_id

oauth2_scheme = HTTPBearer(auto_error=False)

SESSION_PREFIX = "session:"
SESSION_COOKIE = "X-Session-Id"
SESSION_TTL = 7 * 24 * 3600
RENEW_THRESHOLD = 24 * 3600


class UserContext(BaseModel):
    id: int
    username: str
    nickname: Optional[str] = None
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


def _split_authorities(authorities: list) -> tuple[list[str], list[str]]:
    roles = [str(a)[len("ROLE_"):] for a in authorities if str(a).startswith("ROLE_")]
    perms = [str(a) for a in authorities if not str(a).startswith("ROLE_")]
    return roles, perms


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    redis = Depends(get_redis_client),
) -> UserContext:
    if credentials and credentials.credentials.startswith("dhak_"):
        from app.service.api_key_service import ApiKeyService
        user_context = await ApiKeyService.authenticate_by_key(
            request.state.db, credentials.credentials)
        set_current_user_id(user_context.id)
        return user_context

    session_id = request.cookies.get(SESSION_COOKIE) or request.headers.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.ACCESS_UNAUTHORIZED.msg,
        )

    if not redis:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.TOKEN_INVALID.msg,
        )

    session_json = await redis.get(SESSION_PREFIX + session_id)
    if not session_json:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.TOKEN_INVALID.msg,
        )

    session = json.loads(session_json.decode() if isinstance(session_json, bytes) else session_json)

    ttl = await redis.ttl(SESSION_PREFIX + session_id)
    if ttl > 0 and ttl < RENEW_THRESHOLD:
        await redis.expire(SESSION_PREFIX + session_id, SESSION_TTL)

    user_id = session.get("userId")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ResultCode.TOKEN_INVALID.msg,
        )

    authorities = session.get("authorities") or []
    roles, perms = _split_authorities(authorities)

    user_context = UserContext(
        id=int(user_id),
        username=session.get("username", ""),
        nickname=session.get("nickname"),
        dept_id=session.get("deptId"),
        data_scope=session.get("dataScope"),
        roles=roles,
        permissions=perms,
    )

    set_current_user_id(user_context.id)
    return user_context


async def get_current_user_optional(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
) -> Optional[UserContext]:
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None
