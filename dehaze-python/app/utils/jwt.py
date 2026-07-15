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
        nickname: str,
        roles: list[str],
        permissions: list[str],
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        生成访问令牌

        Args:
            user_id: 用户ID
            username: 用户名
            nickname: 昵称
            roles: 角色列表
            permissions: 权限列表
            expires_delta: 过期时间增量，默认使用配置

        Returns:
            JWT Token 字符串
        """
        jti = str(uuid4())
        now = datetime.now(timezone.utc)

        if expires_delta is None:
            expires_delta = timedelta(seconds=settings.JWT_ACCESS_TOKEN_EXPIRES)

        payload = {
            "jti": jti,
            "sub": username,  # sub 统一为 username（与 Java/Go 一致）
            "userId": user_id,  # camelCase（与 Java/Go 一致）
            "username": username,
            "nickname": nickname,
            "authorities": ["ROLE_" + r for r in roles],  # 数组格式（与 Java/Go 一致）
            "permissions": ",".join(permissions),
            "exp": now + expires_delta,
            "iat": now,
            "type": "access",
        }

        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")
