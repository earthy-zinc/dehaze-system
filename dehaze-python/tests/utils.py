"""
测试工具函数

提供测试辅助功能，包括断言工具、数据生成器等
"""

from typing import Any
from uuid import uuid4


def assert_response_success(response: dict) -> None:
    assert response.get("code") == 200, f"Expected code 200, got {response.get('code')}"
    assert "data" in response, "Response missing 'data' field"


def assert_response_error(response: dict, expected_code: int = 500) -> None:
    assert response.get("code") == expected_code, (
        f"Expected code {expected_code}, got {response.get('code')}"
    )
    assert "message" in response, "Response missing 'message' field"


def generate_test_session(
    user_id: int,
    username: str = "testuser",
    roles: list[str] | None = None,
    dept_id: int = 1,
    data_scope: int = 1,
) -> tuple[str, dict]:
    session_id = str(uuid4())
    session_data = {
        "userId": user_id,
        "username": username,
        "nickname": "Test User",
        "deptId": dept_id,
        "dataScope": data_scope,
        "authorities": ["ROLE_" + r for r in (roles or ["USER"])],
    }
    return session_id, session_data


def generate_auth_headers(token: str) -> dict[str, str]:
    """生成认证请求头（Session 模式）"""
    return {"X-Session-Id": token, "Cookie": f"X-Session-Id={token}"}


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
