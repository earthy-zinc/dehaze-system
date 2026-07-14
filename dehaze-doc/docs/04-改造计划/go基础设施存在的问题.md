# dehaze-go 后端特定问题与改进建议

> 文档定位：仅收录 **Go 特有问题**（性能/优化/bug），即由 Go 语言特性或 Gin/GORM 框架特性导致的实现缺陷。
>
> 跨三端的通用问题（架构设计层面 + 模块业务设计层面）已提取到 [通用基础设施问题与改进](./通用基础设施问题与改进.md)。
>
> 核对基准日期：2026-07-13
> 代码版本：dehaze-go（go 1.25.0，Gin v1.11.0，GORM v1.31.0）
> API 测试状态：dehaze-sdk-js `pnpm test:go` 349/349 通过

---

## 一、问题总览

| 严重程度 | 数量 | 典型代表 |
|---------|------|---------|
| HIGH | 9 | GORM 回调未注册、两个冲突的 BaseModel、DataScope 完全失效、Captcha Bug |
| MEDIUM | 9 | 端口 8999 vs 8990、AntiRepeat 覆盖不足、伪策略工厂、HTTP 状态码不一致 |
| LOW | 2 | 空文件残留、ShouldBind 用法 |

> 通用问题（CORS 违法组合、缓存防护未启用、监控端点鉴权、健康检查不完整、任务 DLQ/清理、TraceId context 断裂、数据权限异步失效等）见 [通用基础设施问题与改进](./通用基础设施问题与改进.md)

---

## 二、数据库与 ORM 层（Go/GORM 特有）

### 2.1 [HIGH] GORM 自动填充回调从未注册，create_by / update_by 永远为空

[pkg/database/callback.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/callback.go) 第 103-113 行定义了 `RegisterGormCallbacks(db)`，但 [pkg/database/mysql/client.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/mysql/client.go) 第 86 行的 `initMaster` 只调用了 `RegisterDataScopePlugin(db)`，**未调用** `RegisterGormCallbacks(db)`。全局搜索 `RegisterGormCallbacks` 仅出现在定义处，无任何调用点。`sqlite/client.go`、`postgres/client.go` 同样未调用。

**影响**：所有通过 GORM Create/Update 的记录，`create_by`/`update_by` 字段恒为零值（0），数据审计链断裂，DataScope 的"仅本人数据"权限规则失效。

### 2.2 [HIGH] GormContextMiddleware 未注册，回调拿不到用户上下文

[pkg/database/callback.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/callback.go) 第 93-99 行定义了 `GormContextMiddleware`，但 [internal/app/app.go](file:///E:/DehazeSystem/dehaze-go/internal/app/app.go) 第 41-55 行的全局中间件链未包含它。`autoFillCreateBy`（callback.go 第 164 行）调用 `GetCurrentGinContext()`，未注册时恒返回 nil → 回调直接 return。

**影响**：即使修复 2.1 注册了回调，由于拿不到 gin.Context，`create_by`/`update_by` 仍然填不进去。

### 2.3 [HIGH] AutoFillUserMiddleware 未注册

[pkg/server/gin/middleware/auto_fill_user.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/auto_fill_user.go) 第 19-29 行定义完整，全局搜索 `AutoFillUserMiddleware` 仅出现在定义处，无任何注册调用。

### 2.4 [HIGH] 存在两个冲突的 BaseModel 定义

| 位置 | 字段 | 用途 |
|------|------|------|
| [internal/model/base.go](file:///E:/DehazeSystem/dehaze-go/internal/model/base.go) 第 7-11 行 | `ID` / `CreatedAt` / `UpdatedAt` | **实际被所有实体内嵌** |
| [pkg/database/callback.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/callback.go) 第 15-20 行 | `CreateTime` / `UpdateTime` / `CreateBy` / `UpdateBy` | **形同虚设，无任何实体引用** |

实体内嵌 `model.BaseModel` 后又单独声明 `CreateBy`/`UpdateBy`/`Deleted` 字段。字段命名混乱（`CreatedAt` vs `CreateTime`），反射回调 `FieldByName("CreateBy")` 依赖实体自定义字段而非 BaseModel。

### 2.5 [HIGH] 逻辑删除全靠手动 `deleted = 0`，无 GORM 统一机制

[internal/repository/user/user_repository.go](file:///E:/DehazeSystem/dehaze-go/internal/repository/user/user_repository.go) 中每个查询都手动加 `AND deleted = 0`（第 30、43、54、69、84、99-100、113、180、223、279、364、453、509、545 行），没有使用 `gorm.DeletedAt` 软删除机制。任何遗漏 `deleted = 0` 的查询都会泄露已删除数据。

### 2.6 [HIGH] DataScope 行级权限插件已注册但完全失效

[pkg/database/data_scope.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/data_scope.go) 实现了完整的行级权限插件，`dataScopeCallback`（第 90-142 行）第一行 `c := GetCurrentGinContext()`，若 `c == nil` 则 return。由于 `GormContextMiddleware` 未注册（见 2.2），DataScope **完全不生效**。`DefaultDataScopeConfig`（第 307-326 行）只配置了 `sys_user` 和 `sys_dept` 两张表，其他业务表未配置。

### 2.7 [MEDIUM] FindUserAuthInfo 一次登录触发 4 次 DB 查询

[internal/repository/user/user_repository.go](file:///E:/DehazeSystem/dehaze-go/internal/repository/user/user_repository.go) 第 357-414 行 `FindUserAuthInfo` 串行执行 4 条 SQL：用户基本信息、角色列表、权限列表、数据权限范围。应合并为 1-2 条 JOIN 查询或使用 Redis 缓存。

---

## 三、缓存层（Go 特有）

### 3.1 [HIGH] Captcha CacheStore `UseWithCtx` 逻辑反转 BUG

[pkg/security/captcha.go](file:///E:/DehazeSystem/dehaze-go/pkg/security/captcha.go) 第 44-49 行：

```go
func (rs *CacheStore) UseWithCtx(ctx context.Context) *CacheStore {
    if ctx == nil {   // BUG: 应为 if ctx != nil
        rs.Context = ctx
    }
    return rs
}
```

传入非 nil ctx 时被忽略，传入 nil 时却设置为 nil。验证码 ctx 传递失效，trace_id 链路断裂。

### 3.2 [MEDIUM] `cache.type` 配置字段被忽略

config.yaml 有 `cache.type: redis`，但 [pkg/cache/manager.go](file:///E:/DehazeSystem/dehaze-go/pkg/cache/manager.go) 的 `Init()` 从不读取 `config.Type`，而是分别检查 `Local.Enabled` 和 `Redis.Enabled`。`type` 字段是死配置。

---

## 四、分布式锁（Go 特有）

### 4.1 [HIGH] Lock/Unlock 在业务代码中零调用

[pkg/cache/redis/impl.go](file:///E:/DehazeSystem/dehaze-go/pkg/cache/redis/impl.go) 第 132-160 行实现了 `Lock`/`Unlock`（SetNX + Lua 脚本验证持有者），全局搜索 `.Lock(ctx` 仅命中 multilevel 方法定义，**没有任何 service 或 repository 实际调用**。防重复提交依赖 AntiRepeat 中间件（基于 request body hash + Redis SETEX），未使用分布式锁。

**影响**：任务并发控制无保护（导出任务可能被重复触发），库存/配额类操作无原子性保证。

**改进建议**：参考已有的"分布式锁完整替换方案"（redsync 迁移），在 `TaskExecutor.PublishTask`、`CleanupJob` 等关键路径加锁。

---

## 五、安全层（Go 特有）

### 5.1 [HIGH] Captcha 中间件未注册（冗余代码）

[pkg/server/gin/middleware/captcha.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/captcha.go) 定义了验证码参数提取中间件，全局搜索 `middleware.Captcha` 无任何调用。验证码校验在 service 层 `Login` 方法中通过 `s.VerifyCaptcha()` 直接完成。

### 5.2 [MEDIUM] AntiRepeat 仅在 2 个路由使用

[internal/router/user.go](file:///E:/DehazeSystem/dehaze-go/internal/router/user.go) 第 37、55 行：仅 `sysRoleRouter.POST("")` 和 `sysDeptRouter.POST("")` 使用了 `middleware.AntiRepeat`。其他写操作（用户增删改、数据集操作、文件上传、任务创建、算法导入等）均无防重复保护。
---

## 六、任务/异步层（Go 特有）

### 6.1 [MEDIUM] CleanupJob 多个方法为 TODO 桩

[pkg/job/cleanup_job.go](file:///E:/DehazeSystem/dehaze-go/pkg/job/cleanup_job.go)：

| 方法 | 行号 | 状态 |
|------|------|------|
| `cleanupExpiredTaskCaches` | 202-214 | 仅 `logger.Debug`，无逻辑 |
| `cleanupCompletedTasks` | 217-224 | 同上，仅 Debug 日志 |
| `cleanupFailedDeletions` | 180-192 | 含 `// TODO: 检查文件是否仍然存在`，当前是"随机清理"（`len(filePath)%10 == 0`） |

### 6.2 [MEDIUM] TaskExecutor "工厂"非真正策略模式

[internal/service/task/factory.go](file:///E:/DehazeSystem/dehaze-go/internal/service/task/factory.go) 仅硬编码返回 RabbitMQ 实现，无策略选择机制。注释说"后续可在此扩展更多实现"，但目前是伪工厂。

---

## 七、文件存储层（Go 特有）

### 7.1 [HIGH] MD5 文件去重在 storage 层未实现

[pkg/storage/minio.go](file:///E:/DehazeSystem/dehaze-go/pkg/storage/minio.go) 第 63-74 行 `Upload` 直接 `PutObject`，**无 MD5 计算、无去重查询**。storage 包内全局搜索 "md5" 无命中。[internal/service/file/sys_file.go](file:///E:/DehazeSystem/dehaze-go/internal/service/file/sys_file.go) 的 `SaveItemFile` 在 service 层做了 MD5 秒传，但若调用方绕过 `FileService.SaveItemFile` 直接调用 `storage.Upload`，会产生重复文件。

---

## 八、配置与部署（Go 特有）

### 8.1 [HIGH] 端口与设计不符：8999 vs 8990

[config/config.yaml](file:///E:/DehazeSystem/dehaze-go/config/config.yaml) 第 14 行 `system.port: 8999`，设计文档 [06-部署架构.md](file:///E:/DehazeSystem/dehaze-doc/docs/02-系统架构/06-部署架构.md) 第 37、87、145 行要求 8990。Nginx `/go-api/*` → Go (8990) 的反向代理配置失效。

### 8.2 [MEDIUM] Swagger host 与实际端口不符

[cmd/main.go](file:///E:/DehazeSystem/dehaze-go/cmd/main.go) 第 18 行 `// @host localhost:8080`，应改为 `localhost:8999`（或 8990）。

### 8.3 [MEDIUM] JWT 配置缺少 refresh-token-ttl

[config/config.yaml](file:///E:/DehazeSystem/dehaze-go/config/config.yaml) 第 30-32 行 yaml 中未设置 `refresh-token-ttl`，[pkg/config/options/jwt.go](file:///E:/DehazeSystem/dehaze-go/pkg/config/options/jwt.go) 第 6 行定义了 `RefreshTokenTTL` 但 fallback 到 7 天（魔法数字）。

---

## 九、日志层（Go 特有）

### 9.1 [HIGH] 生产代码中残留 fmt.Println / fmt.Printf

| 文件 | 行号 | 代码 |
|------|------|------|
| [pkg/database/logger.go](file:///E:/DehazeSystem/dehaze-go/pkg/database/logger.go) | 51, 62, 73, 98, 111, 123 | GormLogger 在 `UseZap=false` 分支用 `fmt.Printf` |
| [pkg/logger/cutter.go](file:///E:/DehazeSystem/dehaze-go/pkg/logger/cutter.go) | 103 | `fmt.Printf("清理过期日志失败: %v\n", err)` |
| [pkg/server/gin/middleware/tls.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/tls.go) | 21 | `fmt.Println(err)` |

日志不经过 zap，无法被 trace_id 关联、无法按级别过滤、无法写入日志文件。

---

## 十、目录结构与分层（Go 特有）

### 10.3 [LOW] 空文件残留

[pkg/config/loader.go](file:///E:/DehazeSystem/dehaze-go/pkg/config/loader.go) 和 [pkg/config/watcher.go](file:///E:/DehazeSystem/dehaze-go/pkg/config/watcher.go) 仅含 `package config` 一行，实际逻辑在 [pkg/config/viper.go](file:///E:/DehazeSystem/dehaze-go/pkg/config/viper.go)。

---

## 十一、错误处理（Go 特有）

### 11.1 [MEDIUM] HTTP 状态码使用不一致

[pkg/common/response.go](file:///E:/DehazeSystem/dehaze-go/pkg/common/response.go) 第 28 行 `result()` 恒返回 `http.StatusOK`（200），错误码在 body 中。[pkg/server/gin/middleware/jwt.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/jwt.go) 第 46-52 行 `unauthorized()` 返回 `http.StatusUnauthorized`（401）。业务错误返回 200，认证错误返回 401，不一致。

---

## 十二、context 传播（Go 特有）

### 12.1 [MEDIUM] 多处使用 context.Background() 丢失 trace_id

| 文件 | 行号 | 场景 |
|------|------|------|
| [pkg/server/gin/middleware/anti_repeat.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/anti_repeat.go) | 90 | 防重复提交丢失请求 ctx |
| [pkg/security/jwt.go](file:///E:/DehazeSystem/dehaze-go/pkg/security/jwt.go) | 110 | SetToken 丢失 ctx |
| [pkg/security/permission.go](file:///E:/DehazeSystem/dehaze-go/pkg/security/permission.go) | 326 | GetRolePermissions 丢失 ctx |
| [pkg/security/captcha.go](file:///E:/DehazeSystem/dehaze-go/pkg/security/captcha.go) | 34 | 默认 `context.TODO()` |

应使用 `c.Request.Context()` 而非 `context.Background()`，security 包方法签名增加 `ctx context.Context` 参数。

---

## 十三、修复优先级清单

### P0（阻断性）

| # | 问题 | 文件 |
|---|------|------|
| 1 | 注册 GormContextMiddleware + AutoFillUserMiddleware + RegisterGormCallbacks | app.go, mysql/sqlite/postgres client.go |
| 2 | 修复 Captcha UseWithCtx 逻辑反转 | captcha.go 第 45 行 |
| 3 | 统一 BaseModel 定义，移除冲突 | base.go, callback.go, 所有实体 |

### P1（重要）

| # | 问题 | 文件 |
|---|------|------|
| 4 | 逻辑删除统一为 GORM 机制 | 全部 repository |
| 5 | DataScope 配置扩展 + 启用 | data_scope.go |
| 6 | 端口 8999 vs 8990 确认与同步 | config.yaml, backend.ts |
| 7 | 清理 fmt.Println/fmt.Printf | logger.go, cutter.go, tls.go |
| 8 | Lock/Unlock 业务接入或迁移 redsync | service 层 |
| 9 | MD5 文件去重 storage 层兜底 | minio.go |
| 10 | AntiRepeat 覆盖所有写操作 | internal/router/*.go |

### P2（改进）

| # | 问题 | 说明 |
|---|------|------|
| 11 | FindUserAuthInfo 查询合并 | user_repository.go |
| 12 | context.Background() 清理 | trace_id 贯通 |
| 15 | CleanupJob TODO 桩补全 | cleanup_job.go |
