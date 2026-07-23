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
        dept_id: int | None = None,
        data_scope: int | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        生成访问令牌

        Args:
            user_id: 用户ID
            username: 用户名
            roles: 角色列表
            dept_id: 部门ID
            data_scope: 数据权限范围
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
            "sub": username,
            "userId": user_id,
            "deptId": dept_id,
            "dataScope": data_scope,
            "authorities": ["ROLE_" + r for r in roles],
            "exp": now + expires_delta,
            "iat": now,
        }

        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")
