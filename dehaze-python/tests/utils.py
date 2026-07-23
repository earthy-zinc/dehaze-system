"""
测试工具函数

提供测试辅助功能，包括断言工具、数据生成器等
"""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jose import jwt


def assert_response_success(response: dict) -> None:
    """断言响应成功"""
    assert response.get("code") == 200, f"Expected code 200, got {response.get('code')}"
    assert "data" in response, "Response missing 'data' field"


def assert_response_error(response: dict, expected_code: int = 500) -> None:
    """断言响应错误"""
    assert response.get("code") == expected_code, (
        f"Expected code {expected_code}, got {response.get('code')}"
    )
    assert "message" in response, "Response missing 'message' field"


def generate_test_token(
    user_id: int,
    username: str = "testuser",
    roles: list[str] | None = None,
    dept_id: int | None = 1,
    data_scope: int | None = 1,
    secret_key: str = "test-jwt-secret-key-for-testing-32chars!",
    expires_in: int = 7200,
) -> str:
    """
    生成测试用 JWT Token

    Args:
        user_id: 用户 ID
        username: 用户名
        roles: 角色列表
        dept_id: 部门ID
        data_scope: 数据权限范围
        secret_key: JWT 密钥
        expires_in: 过期时间（秒）

    Returns:
        JWT Token 字符串
    """
    if roles is None:
        roles = ["USER"]
    jti = str(uuid4())
    payload = {
        "jti": jti,
        "sub": username,
        "userId": user_id,
        "deptId": dept_id,
        "dataScope": data_scope,
        "authorities": ["ROLE_" + r for r in roles],
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, secret_key, algorithm="HS256")


def generate_auth_headers(token: str) -> dict[str, str]:
    """生成认证请求头"""
    return {"Authorization": f"Bearer {token}"}


class TestDataFactory:
    """测试数据工厂"""

    @staticmethod
    def create_user_data(
        username: str = "testuser",
        nickname: str = "Test User",
        password: str = "password123",
        **kwargs,
    ) -> dict[str, Any]:
        """创建用户数据"""
        return {
            "username": username,
            "nickname": nickname,
            "password": password,
            "gender": kwargs.get("gender", 1),
            "deptId": kwargs.get("deptId", 1),
            "mobile": kwargs.get("mobile", "13800138000"),
            "email": kwargs.get("email", f"{username}@example.com"),
            "status": kwargs.get("status", 1),
            "roleIds": kwargs.get("roleIds", []),
        }

    @staticmethod
    def create_role_data(
        name: str = "测试角色",
        code: str = "TEST",
        **kwargs,
    ) -> dict[str, Any]:
        """创建角色数据"""
        return {
            "name": name,
            "code": code,
            "sort": kwargs.get("sort", 1),
            "status": kwargs.get("status", 1),
            "dataScope": kwargs.get("dataScope", 1),
        }

    @staticmethod
    def create_menu_data(
        name: str = "测试菜单",
        path: str = "/test",
        **kwargs,
    ) -> dict[str, Any]:
        """创建菜单数据"""
        return {
            "name": name,
            "path": path,
            "component": kwargs.get("component", "Layout"),
            "perms": kwargs.get("perms", "test"),
            "sort": kwargs.get("sort", 1),
            "status": kwargs.get("status", 1),
            "type": kwargs.get("type", 0),
            "parentId": kwargs.get("parentId", 0),
        }

    @staticmethod
    def create_dept_data(
        name: str = "测试部门",
        **kwargs,
    ) -> dict[str, Any]:
        """创建部门数据"""
        return {
            "name": name,
            "parentId": kwargs.get("parentId", 0),
            "sort": kwargs.get("sort", 1),
            "status": kwargs.get("status", 1),
        }
