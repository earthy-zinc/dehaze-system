# DehazeSystem Python 测试框架文档

## 概述

本测试框架为 dehaze-python 项目提供全面的自动化测试支持，基于 **FastAPI + pytest + pytest-asyncio** 构建，支持异步测试、数据库隔离和 Mock 外部依赖。

## 技术栈

- **pytest**: 测试框架
- **pytest-asyncio**: 异步测试支持
- **httpx**: 异步 HTTP 客户端（API 测试）
- **aiosqlite**: SQLite 异步驱动（内存数据库）

## 核心特性

### 1. 异步测试支持

```python
@pytest.mark.asyncio
async def test_login(db_session: AsyncSession):
    result = await AuthService.login(db_session, "user", "pass")
    assert result is not None
```

### 2. 数据库隔离

- 使用 **SQLite 内存数据库** 进行测试，快速且隔离
- 每个测试函数独立事务，测试后自动回滚
- Session 级别创建表，Function 级别隔离数据

### 3. Mock Redis

```python
@pytest.fixture
def mock_redis() -> MockRedis:
    return MockRedis()
```

### 4. Pytest Fixtures

#### 核心 Fixtures

| Fixture | 作用域 | 说明 |
|---------|--------|------|
| `event_loop` | session | 事件循环 |
| `setup_database` | session | 创建数据库表 |
| `db_session` | function | 数据库会话（自动回滚） |
| `mock_redis` | function | Mock Redis 客户端 |
| `client` | function | 异步 HTTP 客户端 |
| `app` | function | FastAPI 应用实例 |

#### 数据 Fixtures

| Fixture | 说明 |
|---------|------|
| `sample_user` | 创建测试用户 |
| `sample_roles` | 创建测试角色（管理员、普通用户） |
| `sample_menu` | 创建测试菜单 |
| `auth_headers` | 认证请求头（含 JWT Token） |

## 项目结构

```text
tests/
├── conftest.py              # pytest 核心配置和共享 fixtures
├── test_config.py           # 配置验证测试
├── utils.py                 # 测试工具函数
├── unit/                    # 单元测试
│   ├── test_auth_service.py
│   └── ...
├── integration/             # 集成测试（待添加）
├── resources/               # 测试资源文件
└── README.md               # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
cd dehaze-python
uv sync --extra test
```

### 2. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定文件
pytest tests/unit/test_auth_service.py

# 运行特定测试类
pytest tests/unit/test_auth_service.py::TestAuthServiceLogin

# 运行特定测试方法
pytest tests/unit/test_auth_service.py::TestAuthServiceLogin::test_login_success

# 显示详细输出
pytest -v

# 显示打印输出
pytest -s

# 运行并显示覆盖率
pytest --cov=app --cov-report=html

# 并行运行
pytest -n auto
```

### 3. 测试标记

```bash
# 只运行单元测试
pytest -m unit

# 只运行 API 测试
pytest -m api

# 跳过慢速测试
pytest -m "not slow"

# 只运行需要数据库的测试
pytest -m requires_db
```

## 编写测试

### 单元测试示例

```python
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.service.auth_service import AuthService


@pytest.mark.unit
@pytest.mark.requires_db
class TestAuthServiceLogin:
    """登录功能测试"""

    @pytest.mark.asyncio
    async def test_login_success(self, db_session: AsyncSession, sample_user: dict):
        """测试登录成功"""
        result = await AuthService.login(
            db=db_session,
            username=sample_user["username"],
            password=sample_user["password"],
        )

        assert result["tokenType"] == "Bearer"
        assert "accessToken" in result
```

### API 测试示例

```python
import pytest


@pytest.mark.api
class TestAuthAPI:
    """认证 API 接口测试"""

    @pytest.mark.asyncio
    async def test_captcha_api(self, client):
        """测试验证码 API"""
        response = await client.get("/api/v1/auth/captcha")
        assert response.status_code == 200

        data = response.json()
        assert data["code"] == 200
```

### 使用 Fixtures

```python
@pytest.mark.asyncio
async def test_with_auth(client, auth_headers):
    """使用认证请求头"""
    response = await client.get("/api/v1/users", headers=auth_headers)
    assert response.status_code == 200
```

## 测试数据工厂

```python
from tests.utils import TestDataFactory

def test_create_user():
    user_data = TestDataFactory.create_user_data(
        username="custom_user",
        nickname="Custom User",
    )
    assert user_data["username"] == "custom_user"
```

## 工具函数

```python
from tests.utils import (
    assert_response_success,
    assert_response_error,
    generate_test_token,
    generate_auth_headers,
)

# 断言响应成功
assert_response_success(response.json())

# 生成测试 Token
token = generate_test_token(user_id=1, username="test")
headers = generate_auth_headers(token)
```

## 最佳实践

### 1. 测试命名

- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试函数：`test_*`

### 2. 测试组织

- 按功能模块组织测试类
- 使用 `@pytest.mark` 标记测试类型
- 单元测试放在 `tests/unit/`
- 集成测试放在 `tests/integration/`

### 3. 异步测试

```python
# 正确：使用 @pytest.mark.asyncio
@pytest.mark.asyncio
async def test_async_operation(db_session):
    result = await some_async_function(db_session)
    assert result is not None

# 错误：忘记标记
async def test_async_operation(db_session):  # 不会正确执行
    ...
```

### 4. 数据隔离

```python
# 正确：使用 fixture，测试后自动回滚
@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession):
    user = SysUser(username="test", ...)
    db_session.add(user)
    await db_session.commit()
    # 测试结束后自动回滚

# 错误：手动创建 session，可能污染数据
async def test_create_user():
    async with SomeSession() as session:
        ...
```

### 5. Mock 外部依赖

```python
# 正确：使用 mock_redis fixture
@pytest.mark.asyncio
async def test_captcha(mock_redis: MockRedis):
    result = await AuthService.get_captcha(mock_redis)
    assert "captchaKey" in result
```

## 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install uv
      run: pip install uv

    - name: Install dependencies
      run: uv sync --extra test

    - name: Run tests
      run: uv run pytest -v --cov=app --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v4
```

## 故障排除

### 常见问题

**1. 异步测试不执行**

```
TypeError: object async_generator can't be used in 'await' expression
```

解决方法：确保添加 `@pytest.mark.asyncio` 装饰器

**2. 数据库连接错误**

```
sqlite3.OperationalError: no such table: sys_user
```

解决方法：确保使用 `db_session` fixture，它会自动创建表

**3. Fixture 未找到**

```
fixture 'xxx' not found
```

解决方法：检查 fixture 名称，确保在 `conftest.py` 中定义

## 扩展测试框架

### 添加新 Fixture

在 `conftest.py` 中添加：

```python
@pytest_asyncio.fixture
async def sample_dataset(db_session: AsyncSession) -> dict:
    """创建测试数据集"""
    from app.models.entity import SysDataset

    dataset = SysDataset(name="Test Dataset", ...)
    db_session.add(dataset)
    await db_session.commit()
    await db_session.refresh(dataset)

    return {"id": dataset.id, "name": dataset.name}
```

### 添加新标记

在 `pytest.ini` 中添加：

```ini
markers =
    slow: 慢速测试
    new_marker: 新标记说明
```

## 参考资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [pytest-asyncio 文档](https://pytest-asyncio.readthedocs.io/)
- [FastAPI 测试文档](https://fastapi.tiangolo.com/tutorial/testing/)
- [httpx 文档](https://www.python-httpx.org/)

## 贡献指南

1. 为新功能编写测试
2. 确保所有测试通过：`pytest`
3. 保持测试覆盖率在 80% 以上
4. 遵循项目代码风格
