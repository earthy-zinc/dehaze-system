"""
JWT Token 工具类

提供 Token 生成功能。Token 解码与用户上下文提取由 dependencies/auth.py 负责。
"""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from jose import jwt

from app.config import settings


class JWTUtils:
    """JWT Token 工具类"""

    @staticmethod
    def create_access_token(
        user_id: int,
        username: str,
        roles: list[str],
        perms: list[str] | None = None,
        dept_id: int | None = None,
        data_scope: int | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        jti = str(uuid4())
        now = datetime.now(timezone.utc)

        if expires_delta is None:
            expires_delta = timedelta(seconds=settings.JWT_ACCESS_TOKEN_EXPIRES)

        authorities = ["ROLE_" + r for r in roles]
        if perms:
            authorities.extend(perms)

        payload = {
            "jti": jti,
            "sub": username,
            "userId": user_id,
            "deptId": dept_id,
            "dataScope": data_scope,
            "authorities": authorities,
            "exp": now + expires_delta,
            "iat": now,
        }

        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    @staticmethod
    def create_refresh_token(
        user_id: int,
        username: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        if expires_delta is None:
            ttl = getattr(settings, "JWT_REFRESH_TOKEN_EXPIRES", 7 * 24 * 3600)
            expires_delta = timedelta(seconds=ttl)

        now = datetime.now(timezone.utc)
        payload = {
            "jti": str(uuid4()),
            "sub": username,
            "userId": user_id,
            "type": "refresh",
            "exp": now + expires_delta,
            "iat": now,
        }
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")
