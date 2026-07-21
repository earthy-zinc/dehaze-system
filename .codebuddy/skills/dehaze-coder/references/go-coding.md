# Go 编码规范（dehaze-go）

基于 dehaze-go 项目实际代码结构提炼的编码约定，所有 Go 代码必须遵守。

> 项目架构与基础设施详见 `dehaze-doc/docs/05-子项目实现/Go后端基础设施文档.md`

---

## 接口定义模式

**必须**在 `interfaces.go` 中定义接口，在 `xxx_service.go` / `xxx_repository.go` 中实现，且返回类型为接口。

```go
// interfaces.go
type IAuthService interface {
    Login(ctx context.Context, req *bo.LoginRequest, clientIP string) (*dto.LoginResult, error)
    Logout(c *gin.Context) error
}

// auth_service.go
type AuthService struct {
    cacheClient types.ICache
    userService userservice.IUserService
}

// 构造函数返回接口类型，而非具体类型
func NewAuthService(cacheClient types.ICache, userService userservice.IUserService) IAuthService {
    return &AuthService{...}
}

// 文件末尾静态断言，防止接口未实现时编译通过
var _ IAuthService = (*AuthService)(nil)
```

---

## 方法注释

公开方法必须有注释，格式：`// FuncName 说明`，并注明对应的 Java 惯例（如有）：

```go
// OkWithData 操作成功，返回数据
// 仿照 Java: return Result.ok(data);
func OkWithData(data interface{}, c *gin.Context) {
    result(SUCCESS, data, c)
}
```

私有辅助方法（如 `getLoginSecurityConfig`）建议也有简短注释说明用途。

---

## 错误处理

**业务错误**使用 `common.BizError`，禁止直接使用 `errors.New` 暴露给调用方。

```go
// 直接构造业务错误
return nil, common.NewBizError(common.PARAM_ERROR, "登录请求不能为空")

// 包装底层错误（保留 cause，但不暴露给前端）
return nil, common.WrapBizError(common.SYSTEM_EXECUTION_ERROR, "生成Token失败", err)

// 错误类型断言
if bizErr, ok := common.AsBizError(err); ok {
    // 使用 bizErr.Code() 和 bizErr.Message()
}
```

- 禁止 `panic`；禁止将底层错误字符串直接拼接给用户
- `gorm.ErrRecordNotFound` 时 Repository 返回 `(nil, nil)`，由上层判断

---

## Repository 层规范

- 构造函数接收 `*gorm.DB`，支持 `WithContext(ctx)` 传递 context
- 分页查询先 `COUNT` 再数据查询（两步）
- 多表联合查询使用 `Scan` 投影到 `read/` 目录的只读结构体
- 事务使用 `db.Transaction(func(tx *gorm.DB) error {...})` 封装，并提供 `Transaction(ctx, fn)` 方法给上层
- 更新时显式指定更新字段（`Select(...).Updates(...)`），避免零值误覆盖

```go
// 指定字段更新
r.db.WithContext(ctx).Model(user).
    Select("nickname", "mobile", "email", "status").
    Updates(user).Error
```

---

## 命名规范

| 场景 | 规范 | 示例 |
|------|------|------|
| 接口 | `I` + 大驼峰 | `IAuthService`, `IUserRepository` |
| 结构体 | 大驼峰 | `AuthService`, `UserRepository` |
| 构造函数 | `New` + 结构体名 | `NewAuthService` |
| model/bo | `动词+Request` / `动词+Form` | `LoginRequest`, `UserForm` |
| model/vo | 大驼峰 + `VO` | `UserInfoVO` |
| model/dto | 大驼峰 + `Result` / `DTO` | `LoginResult`, `CaptchaResult` |
| query | 大驼峰 + `PageQuery` | `UserPageQuery` |
| 常量 | 全大写下划线 | `BlacklistPrefix` |

---

## Context 传递

所有 Repository 和 Service 方法必须接收 `context.Context` 作为第一个参数，通过 `db.WithContext(ctx)` 传递，确保超时、取消信号能正常传播。

Handler 层从 `c.Request.Context()` 或直接传入 `c`（Gin 场景）获取 context。
