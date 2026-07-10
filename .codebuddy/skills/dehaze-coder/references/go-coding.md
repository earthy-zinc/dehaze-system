# Go 编码规范（dehaze-go）

基于 dehaze-go 项目实际代码结构提炼的编码约定，所有 Go 代码必须遵守。

---

## 项目分层与职责

```text
internal/
  api/          Handler 层：HTTP 路由处理，参数绑定，调用 Service
  service/      Service 层：业务逻辑，每个模块一个目录
    [module]/
      interfaces.go     接口定义（IXxxService）
      xxx_service.go    接口实现
  repository/   Repository 层：数据库操作，每个模块一个目录
    [module]/
      interfaces.go     接口定义（IXxxRepository）
      xxx_repository.go 接口实现
  model/        领域模型
    bo/           业务对象（请求参数，对应 Service 入参）
    dto/          传输对象（Service 内部传递的结果）
    vo/           视图对象（Handler 层返回给前端的数据结构）
    query/        分页/筛选查询参数
    read/         只读投影（Repository 层多表查询结果）
    enum/         枚举值定义
pkg/
  common/       通用工具：响应封装、错误码、业务错误
```

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

## 响应封装

统一使用 `pkg/common` 的响应函数，不直接调用 `c.JSON`：

```go
// 成功响应
common.OkWithData(data, c)
common.OkWithMessage("操作成功", c)
common.OkWithDetailed(data, "创建成功", c)

// 失败响应
common.FailWithCode(common.PARAM_ERROR, c)
common.FailWithCodeAndMessage(common.BUSINESS_ERROR, "数据已存在", c)
common.NoAuth("访问未授权", c)  // 返回 HTTP 401
```

响应结构固定为 `{"code": "00000", "data": {...}, "msg": "一切ok"}`。

---

## ResultCode 使用

所有错误码定义在 `pkg/common/result_code.go`，按字母前缀分类：

| 前缀 | 分类 |
|------|------|
| A0xxx | 用户端错误（参数、认证、权限、业务） |
| B0xxx | 系统端错误（超时、资源、限流） |
| C0xxx | 第三方服务错误（DB、缓存、消息、存储） |

新增错误码前先检查是否有可复用的已有码。自定义消息通过 `FailWithCodeAndMessage` 传入，不修改全局 ResultCode 的 Msg。

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

---

## 日志规范

使用 `pkg/logger`（基于 zap）：

```go
logger.Info("用户登录成功", zap.String("username", username), zap.String("clientIP", clientIP))
logger.Warn("登录失败次数超限", zap.String("username", username), zap.Int("failCount", count))
logger.Error("生成Token失败", zap.Error(err))
```

- `Info`：正常业务事件（登录、注销、关键操作）
- `Warn`：可恢复的异常（失败次数超限、缓存操作失败）
- `Error`：不可恢复的异常（Token 生成失败、核心流程出错）
- 禁止在日志中输出密码、Token 完整内容等敏感信息
