# DehazeSystem Python 测试框架文档

## 概述

本测试框架为 dehaze-python 项目提供全面的自动化测试支持，基于 pytest 构建，支持多种数据库后端，具有良好的测试隔离和可扩展性。

## 核心特性

### 1. 多数据库支持

测试框架支持三种数据库模式，通过环境变量 `TEST_DATABASE_TYPE` 配置：

- **SQLite 内存数据库**（默认）：快速、轻量级，适合本地开发
- **MySQL 测试库**：与生产环境一致，用于集成测试
- **PostgreSQL 测试库**：跨数据库兼容性测试

```bash
# 使用 SQLite（默认）
pytest

# 使用 MySQL
TEST_DATABASE_TYPE=mysql pytest

# 使用 PostgreSQL
TEST_DATABASE_TYPE=postgresql pytest
```

### 2. 测试隔离机制

- **Function 级别隔离**：每个测试函数拥有独立的数据库会话
- **自动清理**：测试前创建表，测试后自动删除
- **无状态污染**：测试之间完全隔离，互不影响

### 3. Pytest Fixtures

#### 核心 Fixtures

- `app`: Session 级别的 Flask 应用实例
- `db_session`: Function 级别的数据库会话
- `client`: Flask 测试客户端
- `runner`: Flask CLI runner

#### 数据 Fixtures

- `sample_roles`: 创建测试角色（管理员、普通用户）
- `sample_user`: 创建测试用户

## 项目结构

```text
tests/
├── conftest.py              # pytest 核心配置和共享 fixtures
├── pytest.ini               # pytest 配置文件（项目根目录）
├── unit/                    # 单元测试
│   ├── __init__.py
│   └── test_user_service.py
├── integration/             # 集成测试（待添加）
└── README.md               # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv 安装依赖（推荐）
cd dehaze-python
uv sync --extra test

# 或使用 pip
pip install -e ".[test]"
```

### 2. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/unit/test_user_service.py

# 运行特定测试函数
pytest tests/unit/test_user_service.py::test_create_user

# 显示详细输出
pytest -v

# 显示测试覆盖率
pytest --cov=app --cov-report=html

# 并行运行测试（需要 pytest-xdist）
pytest -n auto
```

### 3. 测试标记

使用标记筛选测试：

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 跳过慢速测试
pytest -m "not slow"

# 只运行需要数据库的测试
pytest -m requires_db
```

## 编写测试

### 基本测试示例

```python
import pytest
from app.service.user import UserService

@pytest.mark.unit
class TestUserService:
    """用户服务测试"""
    
    def test_create_user(self, db_session):
        """测试创建用户"""
        user = UserService.create_user(
            username='testuser',
            password='password123',
            nickname='Test User'
        )
        
        assert user is not None
        assert user.username == 'testuser'
        assert user.nickname == 'Test User'
    
    def test_get_user_by_username(self, db_session, sample_user):
        """测试根据用户名获取用户"""
        user = UserService.get_user_by_username('testuser')
        
        assert user is not None
        assert user.id == sample_user.id
```

### 使用 Fixtures

```python
def test_with_roles(self, db_session, sample_roles):
    """使用角色 fixture 的测试"""
    admin_role = sample_roles['admin']
    user_role = sample_roles['user']
    
    assert admin_role.code == 'ADMIN'
    assert user_role.code == 'USER'

def test_with_client(self, client):
    """使用测试客户端"""
    response = client.get('/api/v1/users')
    assert response.status_code == 200
```

## 数据库配置

### MySQL 测试数据库

```python
# config.py 中的配置
SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@localhost/dehaze_test?charset=utf8"
```

**注意**：需要手动创建 MySQL 测试数据库：

```sql
CREATE DATABASE dehaze_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### PostgreSQL 测试数据库

```python
# config.py 中的配置
SQLALCHEMY_DATABASE_URI = "postgresql://root:123456@localhost/dehaze_test"
```

**注意**：PostgreSQL 测试数据库会自动创建和删除，无需手动操作。

## 最佳实践

### 1. 测试命名

- 测试文件：`test_*.py`
- 测试类：`Test*`
- 测试函数：`test_*`

### 2. 测试组织

- 按功能模块组织测试
- 单元测试放在 `tests/unit/`
- 集成测试放在 `tests/integration/`

### 3. 使用 Fixtures

- 尽量使用共享 fixtures 减少重复代码
- 为复杂场景创建自定义 fixtures
- 保持 fixtures 简单和可复用

### 4. 测试隔离

- 不依赖测试执行顺序
- 每个测试应该独立运行
- 使用 fixtures 设置测试数据

### 5. 断言清晰

```python
# 好的断言
assert user.username == 'testuser', "用户名应该是 testuser"

# 避免复杂的断言逻辑
assert user is not None
assert user.status == 1
```

## 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mysql:
        image: mysql:8.0
        env:
          MYSQL_ROOT_PASSWORD: 123456
          MYSQL_DATABASE: dehaze_test
        ports:
          - 3306:3306
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --extra test
    
    - name: Run tests with SQLite
      run: pytest
    
    - name: Run tests with MySQL
      run: TEST_DATABASE_TYPE=mysql pytest
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## 故障排除

### 常见问题

**1. 数据库连接失败**

```
OperationalError: (2003, "Can't connect to MySQL server...")
```

解决方法：

- 确保数据库服务正在运行
- 检查连接凭据是否正确
- 验证测试数据库是否存在

**2. SQLite 多线程问题**

```
sqlite3.ProgrammingError: SQLite objects created in a thread...
```

解决方法：已在 `TestingConfig` 中配置 `check_same_thread=False`

**3. 表不存在错误**

```
OperationalError: (1146, "Table 'dehaze_test.sys_user' doesn't exist")
```

解决方法：

- 确保使用了 `db_session` fixture
- 检查 `mysql.create_all()` 是否正确执行

## 扩展测试框架

### 添加新的 Fixtures

在 `conftest.py` 中添加：

```python
@pytest.fixture
def sample_algorithm(db_session):
    """创建测试算法"""
    algorithm = SysAlgorithm(
        name='Test Algorithm',
        type='dehazing',
        status=1
    )
    db_session.add(algorithm)
    db_session.commit()
    return algorithm
```

### 添加新的测试标记

在 `pytest.ini` 中添加：

```ini
markers =
    api: API 接口测试
    model: 模型层测试
```

## 参考资源

- [Pytest 官方文档](https://docs.pytest.org/)
- [Flask Testing 文档](https://flask.palletsprojects.com/testing/)
- [pytest-flask 插件](https://pytest-flask.readthedocs.io/)

## 贡献指南

1. 为新功能编写测试
2. 确保所有测试通过
3. 保持测试覆盖率在 80% 以上
4. 遵循项目代码风格

## 许可证

与主项目相同
