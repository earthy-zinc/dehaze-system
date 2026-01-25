# Schema 模型组织规范

## 目录结构

```
app/models/schema/
├── __init__.py          # 统一导出所有 Schema 模型
├── common.py            # 公共基础模型（分页、选项、统一响应等）
├── role.py              # 角色模块 ✅
├── user.py              # 用户模块 ✅
├── menu.py              # 菜单模块 ✅
├── dept.py              # 部门模块 ✅
├── dict.py              # 字典模块 ✅
├── dataset.py           # 数据集模块 ✅
├── algorithm.py         # 算法模块 ✅
├── file.py              # 文件模块 ✅
├── task.py              # 任务模块 ✅
└── ...                  # 其他模块
```

---

## 执行计划

### 阶段 1: User 模块改造

**Schema 文件**: `app/models/schema/user.py`

| 类型 | 模型名 | 说明 |
|------|--------|------|
| Query | `UserPageQuery` | 用户分页查询（keywords, status, deptId, startTime, endTime） |
| Query | `UserStatusQuery` | 状态修改查询参数 |
| Query | `UserPasswordQuery` | 密码修改参数 |
| Path | `UserIdPath` | 单个用户ID路径 |
| Path | `UserIdsPath` | 批量用户ID路径 |
| Body | `LoginForm` | 登录表单（username, password） |
| Body | `RegisterForm` | 注册表单 |
| Body | `UserForm` | 用户表单（新增/编辑） |
| Body | `PasswordForm` | 密码修改表单 |
| VO | `LoginVO` | 登录响应（token, user） |
| VO | `UserInfoVO` | 当前用户信息 |
| VO | `UserPageVO` | 用户分页项 |
| VO | `UserFormVO` | 用户表单数据 |

**路由改造**: `app/route/user.py`
- Blueprint → APIBlueprint
- @swag_from → Pydantic 类型注解
- request.get_json() → body: XxxForm
- request.args.get() → query: XxxQuery

### 阶段 2: Menu 模块改造

**Schema 文件**: `app/models/schema/menu.py`

| 类型 | 模型名 | 说明 |
|------|--------|------|
| Query | `MenuQuery` | 菜单列表查询（keywords, status） |
| Query | `MenuVisibleQuery` | 可见状态查询 |
| Path | `MenuIdPath` | 菜单ID路径 |
| Body | `MenuForm` | 菜单表单 |
| VO | `MenuVO` | 菜单视图对象（含 children 树结构） |
| VO | `RouteVO` | 路由视图对象 |
| VO | `MenuOptionVO` | 菜单下拉选项 |

**路由改造**: `app/route/menu.py`
- 同 User 模块改造方式

---

## 模型分类与命名规范

### 1. 查询参数模型（Query）
用于 URL 查询参数，继承自 `BasePageQuery` 或 `BaseModel`

**命名规范**: `{模块名}{功能}Query`

```python
class UserPageQuery(BasePageQuery):
    """用户分页查询参数"""
    keywords: Optional[str] = Field(default=None, description="关键字(用户名/昵称/手机号)")
    status: Optional[int] = Field(default=None, description="用户状态")
    deptId: Optional[int] = Field(default=None, description="部门ID")
```

### 2. 路径参数模型（Path）
用于 URL 路径参数（如 `/api/v1/users/{user_id}`）

**命名规范**: `{模块名}{参数名}Path`

```python
class UserIdPath(BaseModel):
    """用户ID路径参数"""
    user_id: int = Field(..., description="用户ID")
```

### 3. 请求体模型（Form/Body）
用于 POST/PUT 请求体

**命名规范**: `{模块名}Form` 或 `{模块名}{操作}Body`

```python
class UserForm(BaseModel):
    """用户表单"""
    username: str = Field(..., min_length=1, description="用户名")
    nickname: str = Field(..., min_length=1, description="昵称")
    roleIds: List[int] = Field(..., min_length=1, description="角色ID列表")
```

### 4. 响应模型（VO）
用于接口响应数据

**命名规范**: `{模块名}{功能}VO`

```python
class UserPageVO(BaseModel):
    """用户分页VO"""
    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: str = Field(description="昵称")
```

---

## 模型组织原则

### ✅ 正确做法

1. **所有 Schema 模型定义在 `app/models/schema/` 目录下**
2. **按模块分文件**（一个模块一个文件）
3. **在 `__init__.py` 中统一导出**
4. **路由文件只导入和使用，不定义模型**

```python
# ✅ app/route/user.py
from app.models.schema.user import (
    UserPageQuery, UserIdPath, UserForm, UserPageVO
)

@user_blueprint.get("/page")
def get_user_page(query: UserPageQuery):
    ...
```

### ❌ 错误做法

```python
# ❌ 在路由文件中定义模型或使用 @swag_from
from pydantic import BaseModel
from flasgger import swag_from

@swag_from({...})  # 不应再使用
def get_user_page():
    ...
```

---

## 特殊类型处理

### 列表类型请求体
使用 `RootModel` 包装

```python
from pydantic import RootModel

class RoleIdsBody(RootModel[List[int]]):
    """角色ID列表请求体"""
    root: List[int] = Field(..., description="角色ID列表")

# 使用时
def update_roles(body: RoleIdsBody):
    role_ids = body.root  # 通过 .root 访问
```

### 字段别名（驼峰/下划线转换）
```python
class UserForm(BaseModel):
    dept_id: Optional[int] = Field(default=None, alias="deptId")
    role_ids: List[int] = Field(..., alias="roleIds")
    
    model_config = ConfigDict(populate_by_name=True)
```

### 树形结构（自引用）
```python
from typing import Optional, List, ForwardRef

class MenuVO(BaseModel):
    id: int
    name: str
    children: Optional[List["MenuVO"]] = None

MenuVO.model_rebuild()  # 必须调用以解析自引用
```

---

## 迁移清单

### 已完成 ✅
- [x] `role.py` - 角色管理模块
- [x] `user.py` - 用户管理模块
- [x] `menu.py` - 菜单管理模块
- [x] `dept.py` - 部门管理模块
- [x] `dict.py` - 字典管理模块
- [x] `dataset.py` - 数据集管理模块
- [x] `algorithm.py` - 算法管理模块
- [x] `file.py` - 文件管理模块
- [x] `task.py` - 任务管理模块

### 待迁移 ⏳
（无）

---

## 收益

1. **代码组织清晰** - 模型定义与业务逻辑分离
2. **便于复用** - 模型可在多个路由中共享
3. **易于维护** - 统一管理，修改方便
4. **自动文档** - flask-openapi3 根据模型自动生成 Swagger 文档
5. **类型安全** - 静态类型检查与运行时验证
6. **参数校验** - Pydantic 自动校验，减少手动检查代码
