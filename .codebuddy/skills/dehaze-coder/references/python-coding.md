# Python 编码规范（dehaze-python）

基于 dehaze-python 项目实际代码结构提炼的编码约定，所有 Python 代码必须遵守。

> 项目架构与基础设施详见 `dehaze-doc/docs/04-项目实现/后端/05-Python算法服务架构文档.md`

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
        return await user_repository.get_page(db, page=page, page_size=page_size, keywords=keywords)
```

- 所有公开方法必须有类型注解（参数和返回值）
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

## 配置访问

统一从 `app/config.py` 导入 `settings`，禁止直接读取环境变量：

```python
from app.config import settings

if settings.DEBUG:
    ...
ttl = settings.JWT_SECRET_KEY
min_len = settings.PASSWORD_MIN_LENGTH
```
