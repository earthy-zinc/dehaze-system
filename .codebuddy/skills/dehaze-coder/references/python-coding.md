# Python 编码规范（dehaze-python）

基于 dehaze-python 项目实际代码结构提炼的编码约定，所有 Python 代码必须遵守。

---

## 技术栈

- **Web 框架**：FastAPI（异步）
- **ORM**：SQLAlchemy 2.0（异步 `AsyncSession`，使用 `select()` / `update()` / `delete()` 语句）
- **数据模型**：Pydantic v2（Schema 校验）+ SQLAlchemy Declarative（Entity）
- **配置**：pydantic-settings `BaseSettings`，从 `app/config.py` 导入 `settings`
- **Python 版本**：3.10+（使用 `X | Y` 联合类型语法）

---

## 项目分层与职责

```text
app/
  router/           HTTP 路由层（FastAPI APIRouter）
  service/          业务逻辑层（静态方法 class，不持有状态）
  repository/       数据访问层（实例方法 class，继承 BaseRepository）
  models/
    entity/         SQLAlchemy ORM 实体（映射数据库表）
    schema/         Pydantic Schema（请求/响应数据结构）
  core/
    code.py         ResultCode 枚举
    result.py       Result[T] 泛型 + success/error/warning 函数
    exceptions.py   BusinessException + 全局异常处理注册
  dependencies/
    auth.py         JWT 解码、UserContext、get_current_user
    redis.py        Redis 客户端依赖
  infrastructure/   基础设施（对象存储、任务队列等）
  middleware/       请求中间件
```

---

## Router 层规范

Router 文件只做：参数声明、权限守卫、调用 Service、封装响应。**禁止**在 router 层写业务逻辑。

```python
router = APIRouter(prefix="/api/v1/users", tags=["用户管理"])

@router.get("/page", summary="获取用户分页列表", response_model=Result[PageResult[UserPageVO]])
async def get_user_page(
    pageNum: int = Query(default=1, ge=1, description="页码"),
    pageSize: int = Query(default=10, ge=1, le=100, description="每页记录数"),
    keywords: Optional[str] = Query(default=None, description="关键词"),
    db: AsyncSession = Depends(get_db),
    user: UserContext = Depends(get_current_user),   # 必须鉴权
):
    data, total = await UserService.get_user_list(db, page=pageNum, ...)
    return success({"list": data, "total": total, "pageNum": pageNum, "pageSize": pageSize})
```

- Query 参数必须有 `description`，数值类型必须有 `ge`/`le` 约束
- 需要权限控制时使用 `@require_permission("sys:xxx:yyy")` 装饰器
- `response_model` 指定为 `Result[具体类型]`

---

## Service 层规范

Service 以 **静态方法 class** 形式组织，不持有实例状态：

```python
class UserService:
    """用户服务（异步版本）"""

    @staticmethod
    async def get_user_list(
        db: AsyncSession,
        page: int,
        page_size: int,
        keywords: str | None = None,
    ) -> tuple[list[SysUser], int]:
        """获取用户列表

        Args:
            db: 数据库会话
            page: 页码（从 1 开始）
            page_size: 每页记录数

        Returns:
            (用户列表, 总数)
        """
        return await user_repository.get_page(db, page=page, page_size=page_size, keywords=keywords)
```

- 所有公开方法必须有类型注解（参数和返回值）
- 文档字符串使用 Google 风格（Args/Returns/Raises）
- 业务校验失败时抛出 `BusinessException`，不返回错误响应

---

## Repository 层规范

Repository 继承 `BaseRepository[T]`，使用 SQLAlchemy 2.0 异步 API：

```python
class UserRepository(BaseRepository[SysUser]):
    model = SysUser

    async def get_by_username(self, db: AsyncSession, username: str) -> SysUser | None:
        stmt = select(SysUser).where(
            SysUser.username == username,
            SysUser.deleted == 0,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()
```

- 模糊查询使用 `escape_like(keyword)` 函数处理特殊字符
- 关联查询使用 `join()`，避免 N+1（不在循环内发起 db 查询）
- `scalar_one_or_none()` 查单条，`scalars().all()` 查列表
- Repository 实例在 `app/repository/` 目录末尾初始化为模块级单例：`user_repository = UserRepository()`

---

## 异常处理

**业务异常**使用 `BusinessException`，全局 exception handler 自动转换为标准响应：

```python
from app.core.exceptions import BusinessException
from app.core.code import ResultCode

# 使用 ResultCode 枚举（推荐，有明确错误码）
raise BusinessException(ResultCode.DATA_EXISTS, "用户名已存在")

# 使用字符串（快速抛错，使用 SYSTEM_EXECUTION_ERROR 错误码）
raise BusinessException("用户名已存在")
```

- 禁止在 router 层 try-catch 后返回自定义 dict，统一由全局 handler 处理
- `SQLAlchemyError` 由全局 handler 捕获，返回 `DATABASE_ERROR`；不需要在 service 层 catch

---

## 响应封装

统一使用 `app/core/result.py` 的工具函数：

```python
from app.core.result import Result, success, error, warning

# 成功响应（data 可以是 dict、Pydantic model 或 None）
return success({"id": user.id, "username": user.username})
return success(user_vo, msg="创建成功")

# 失败响应（直接返回，不抛异常）
return error("仅支持 xlsx 格式", code="B0001")

# 使用 ResultCode（返回 warning 级别的业务失败）
return warning(ResultCode.DATA_EXISTS)
```

`Result[T]` 的结构固定为 `{"code": "00000", "msg": "一切ok", "data": ...}`。

---

## 依赖注入

```python
# 数据库会话（必须在每个有 db 操作的路由中注入）
db: AsyncSession = Depends(get_db)

# 当前用户（必须鉴权的接口）
user: UserContext = Depends(get_current_user)

# 可选鉴权（公开接口但登录用户有额外权限）
user: Optional[UserContext] = Depends(get_current_user_optional)
```

`UserContext` 的 `is_root` 属性判断是否为超级管理员（`ROOT` in roles）。

---

## 类型注解

- 所有公开函数必须有参数和返回值类型注解
- 使用 Python 3.10+ 语法：`X | Y` 代替 `Union[X, Y]`，`X | None` 代替 `Optional[X]`
- Pydantic Schema 中必填字段直接声明，可选字段使用 `field: Type | None = None`
- 避免使用 `Any` 类型，优先用具体类型或泛型

```python
# 推荐
async def get_user(db: AsyncSession, user_id: int) -> SysUser | None: ...

# 不推荐
async def get_user(db, user_id) -> Any: ...
```

---

## 异步规范

- 所有 I/O 操作（db、Redis、HTTP 调用、文件读写）必须使用 `async/await`
- 禁止在 async 函数中使用同步阻塞调用（`time.sleep`、`requests.get`、`open()` 读大文件）
- 长耗时计算（去雾算法、图像处理）放到 `asyncio.get_event_loop().run_in_executor()` 或独立 task queue

---

## 命名规范

| 场景 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | `snake_case` | `user_service.py`, `auth_repository.py` |
| 类名 | 大驼峰 | `UserService`, `UserRepository` |
| 函数/方法 | `snake_case` | `get_user_list`, `create_user_with_roles` |
| Pydantic Schema | 大驼峰 + 用途后缀 | `UserPageVO`, `UserForm`, `LoginRequest` |
| SQLAlchemy Entity | 大驼峰 | `SysUser`, `SysRole` |
| 常量 | 全大写下划线 | `MAX_PAGE_SIZE = 100` |
| 私有函数 | `_` 前缀 | `_extract_permissions`, `_get_role_deleted_column` |

---

## 配置访问

统一从 `app/config.py` 导入 `settings`，禁止直接读取环境变量：

```python
from app.config import settings

# 使用
if settings.DEBUG:
    ...
ttl = settings.JWT_SECRET_KEY
min_len = settings.PASSWORD_MIN_LENGTH
```
