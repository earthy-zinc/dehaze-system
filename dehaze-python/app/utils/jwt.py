"""
JWT Token 工具类

提供 Token 生成、验证、解码等功能
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import JWTError, jwt

from app.config import settings


class TokenData:
    """Token 数据类"""

    def __init__(
        self,
        user_id: int,
        username: str,
        nickname: str,
        roles: list[str],
        permissions: list[str],
        jti: str | None = None,
    ):
        self.user_id = user_id
        self.username = username
        self.nickname = nickname
        self.roles = roles
        self.permissions = permissions
        self.jti = jti


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
            "sub": str(user_id),
            "user_id": user_id,
            "username": username,
            "nickname": nickname,
            "roles": ",".join(roles),
            "permissions": ",".join(permissions),
            "exp": now + expires_delta,
            "iat": now,
            "type": "access",
        }

        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    @staticmethod
    def create_refresh_token(
        user_id: int,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        生成刷新令牌

        Args:
            user_id: 用户ID
            expires_delta: 过期时间增量，默认使用配置

        Returns:
            JWT Token 字符串
        """
        jti = str(uuid4())
        now = datetime.now(timezone.utc)

        if expires_delta is None:
            expires_delta = timedelta(seconds=settings.JWT_REFRESH_TOKEN_EXPIRES)

        payload = {
            "jti": jti,
            "sub": str(user_id),
            "user_id": user_id,
            "exp": now + expires_delta,
            "iat": now,
            "type": "refresh",
        }

        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm="HS256")

    @staticmethod
    def decode_token(token: str) -> dict[str, Any] | None:
        """
        解码 Token

        Args:
            token: JWT Token 字符串

        Returns:
            解码后的 payload，解码失败返回 None
        """
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=["HS256"],
            )
            return payload
        except JWTError:
            return None

    @staticmethod
    def extract_token_data(token: str) -> TokenData | None:
        """
        从 Token 中提取用户数据

        Args:
            token: JWT Token 字符串

        Returns:
            TokenData 对象，提取失败返回 None
        """
        payload = JWTUtils.decode_token(token)
        if not payload:
            return None

        try:
            roles_str = payload.get("roles", "")
            permissions_str = payload.get("permissions", "")

            return TokenData(
                user_id=int(payload["user_id"]),
                username=payload["username"],
                nickname=payload.get("nickname", ""),
                roles=roles_str.split(",") if roles_str else [],
                permissions=permissions_str.split(",") if permissions_str else [],
                jti=payload.get("jti"),
            )
        except (KeyError, ValueError, TypeError):
            return None

    @staticmethod
    def get_jti(token: str) -> str | None:
        """
        获取 Token 的 JTI（JWT ID）

        Args:
            token: JWT Token 字符串

        Returns:
            JTI 字符串，获取失败返回 None
        """
        payload = JWTUtils.decode_token(token)
        return payload.get("jti") if payload else None

    @staticmethod
    def get_user_id(token: str) -> int | None:
        """
        获取 Token 中的用户ID

        Args:
            token: JWT Token 字符串

        Returns:
            用户ID，获取失败返回 None
        """
        payload = JWTUtils.decode_token(token)
        if payload and "user_id" in payload:
            try:
                return int(payload["user_id"])
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def is_expired(token: str) -> bool:
        """
        检查 Token 是否过期

        Args:
            token: JWT Token 字符串

        Returns:
            是否过期
        """
        payload = JWTUtils.decode_token(token)
        if not payload:
            return True

        exp = payload.get("exp")
        if not exp:
            return True

        return datetime.now(timezone.utc).timestamp() > exp

    @staticmethod
    def is_refresh_token(token: str) -> bool:
        """
        检查是否为刷新令牌

        Args:
            token: JWT Token 字符串

        Returns:
            是否为刷新令牌
        """
        payload = JWTUtils.decode_token(token)
        return payload.get("type") == "refresh" if payload else False


# 便捷函数
def create_tokens(
    user_id: int,
    username: str,
    nickname: str,
    roles: list[str],
    permissions: list[str],
) -> dict[str, str]:
    """
    创建访问令牌和刷新令牌

    Args:
        user_id: 用户ID
        username: 用户名
        nickname: 昵称
        roles: 角色列表
        permissions: 权限列表

    Returns:
        {"accessToken": ..., "refreshToken": ...}
    """
    access_token = JWTUtils.create_access_token(
        user_id=user_id,
        username=username,
        nickname=nickname,
        roles=roles,
        permissions=permissions,
    )
    refresh_token = JWTUtils.create_refresh_token(user_id=user_id)

    return {
        "accessToken": access_token,
        "refreshToken": refresh_token,
    }
