"""
pytest 配置和共享 fixtures

基于 FastAPI + pytest-asyncio 的测试框架
"""

import os
import sys
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置测试环境变量
os.environ["APP_ENV"] = "testing"
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-for-testing-32chars!"
os.environ["DEHAZE_PASSWORD"] = "test_password"

from app.main import app as fastapi_app
from tests.test_models import MockBase, MockRole, MockUser, MockUserRole


# ==================== 数据库引擎 ====================

# 使用 SQLite 内存数据库进行测试（快速、隔离）
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    future=True,
)

test_session_factory = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ==================== Mock Redis ====================

class MockRedis:
    """Mock Redis 客户端"""

    def __init__(self):
        self._data: dict[str, str] = {}

    async def get(self, key: str) -> bytes | None:
        value = self._data.get(key)
        return value.encode() if value else None

    async def set(self, key: str, value: str, *args, **kwargs) -> bool:
        self._data[key] = value
        return True

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self._data[key] = value
        return True

    async def delete(self, key: str) -> int:
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    async def exists(self, key: str) -> bool:
        return key in self._data

    async def expire(self, key: str, ttl: int) -> bool:
        return key in self._data

    async def ttl(self, key: str) -> int:
        return -1 if key not in self._data else 3600

    async def incr(self, key: str) -> int:
        if key not in self._data:
            self._data[key] = "0"
        self._data[key] = str(int(self._data[key]) + 1)
        return int(self._data[key])

    async def close(self) -> None:
        pass


# ==================== Fixtures ====================

@pytest_asyncio.fixture(loop_scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Function 级别：提供独立的数据库会话
    每个测试创建新表，测试后自动清理
    """
    # 每个测试创建表（使用测试模型）
    async with test_engine.begin() as conn:
        await conn.run_sync(MockBase.metadata.create_all)

    async with test_session_factory() as session:
        yield session

    # 测试后清理
    async with test_engine.begin() as conn:
        await conn.run_sync(MockBase.metadata.drop_all)




@pytest.fixture
def mock_redis() -> MockRedis:
    """Mock Redis 客户端"""
    return MockRedis()


@pytest_asyncio.fixture
async def client(mock_redis: MockRedis) -> AsyncGenerator[AsyncClient, None]:
    """
    异步测试客户端，重写 Redis 依赖为 MockRedis
    """
    from app.dependencies.redis import get_redis_client, get_redis

    async def _override_redis():
        return mock_redis

    fastapi_app.dependency_overrides[get_redis_client] = _override_redis
    fastapi_app.dependency_overrides[get_redis] = _override_redis

    async with AsyncClient(
        transport=ASGITransport(app=fastapi_app),
        base_url="http://test",
    ) as client:
        yield client

    fastapi_app.dependency_overrides.pop(get_redis_client, None)


@pytest.fixture
def app():
    """FastAPI 应用实例"""
    return fastapi_app


# ==================== 数据 Fixtures ====================

@pytest_asyncio.fixture
async def sample_user(db_session: AsyncSession) -> dict:
    """创建测试用户"""
    from app.utils.password import hash_password_async

    hashed_password = await hash_password_async("password123")

    user = MockUser(
        username="testuser",
        nickname="Test User",
        password=hashed_password,
        gender=1,
        dept_id=1,
        mobile="13800138000",
        email="test@example.com",
        status=1,
        deleted=0,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname,
        "password": "password123",  # 明文密码，用于测试
    }


@pytest_asyncio.fixture
async def sample_roles(db_session: AsyncSession) -> dict:
    """创建测试角色"""
    admin_role = MockRole(
        name="管理员",
        code="ADMIN",
        sort=1,
        status=1,
        data_scope=1,
        deleted=0,
    )
    user_role = MockRole(
        name="普通用户",
        code="USER",
        sort=2,
        status=1,
        data_scope=2,
        deleted=0,
    )

    db_session.add_all([admin_role, user_role])
    await db_session.commit()
    await db_session.refresh(admin_role)
    await db_session.refresh(user_role)

    return {"admin": admin_role, "user": user_role}


@pytest_asyncio.fixture
async def sample_menu(db_session: AsyncSession) -> dict:
    """创建测试菜单"""
    # 简化的菜单测试数据
    return {"menu": {"id": 1, "name": "系统管理", "path": "/system"}}


@pytest_asyncio.fixture
async def auth_headers(sample_user: dict, mock_redis: MockRedis) -> dict:
    """获取认证请求头（Session 模式）"""
    import json
    from uuid import uuid4

    session_id = str(uuid4())
    session_data = json.dumps({
        "userId": sample_user["id"],
        "username": sample_user["username"],
        "nickname": sample_user.get("nickname", ""),
        "deptId": 1,
        "dataScope": 1,
        "authorities": ["ROLE_USER"],
    })
    await mock_redis.set("session:" + session_id, session_data)

    return {"X-Session-Id": session_id, "Cookie": f"X-Session-Id={session_id}"}


# ==================== 工具函数 ====================

def create_test_user_dict(
    username: str = "testuser",
    nickname: str = "Test User",
    **kwargs,
) -> dict:
    """创建测试用户数据字典"""
    return {
        "username": username,
        "nickname": nickname,
        "gender": 1,
        "deptId": 1,
        "mobile": "13800138000",
        "email": f"{username}@example.com",
        **kwargs,
    }
