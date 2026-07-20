# dehaze-go 后端特定问题与改进建议

> 文档定位：仅收录 **Go 特有问题**（性能/优化/bug），即由 Go 语言特性或 Gin/GORM 框架特性导致的实现缺陷。
>
> 核对基准日期：2026-07-20
> 代码版本：dehaze-go（go 1.25.0，Gin v1.11.0，GORM v1.31.0）
> API 测试状态：dehaze-sdk-js `pnpm test:go` 349/349 通过

---

## 一、问题总览

| 严重程度 | 数量 | 典型代表 |
|---------|------|---------|
| MEDIUM | 2 | Captcha 死代码、端口与设计文档不符 |

---

## 二、安全层（Go 特有）

### 2.1 [MEDIUM] Captcha 中间件为死代码，仅加 Deprecated 注释未删除

[pkg/server/gin/middleware/captcha.go](file:///E:/DehazeSystem/dehaze-go/pkg/server/gin/middleware/captcha.go) 第 21 行定义的 `Captcha(config CaptchaConfig)` 函数，全局搜索 `middleware.Captcha` **零调用**。验证码校验实际在 Service 层 `AuthService.Login → VerifyCaptcha` 完成，无需中间件提取参数。

文件第 18-20 行虽已加 `// Deprecated:` 注释：

```go
// Deprecated: 此中间件未在任何路由中注册。验证码校验由 Service 层
// AuthService.Login → VerifyCaptcha 直接完成，无需中间件提取参数。
// 保留仅供参考，后续版本将移除。
```

**影响**：死代码长期残留会增加维护负担，且 `// Deprecated: 保留仅供参考` 的做法违反"禁止兼容历史烂逻辑"原则——要么删除，要么真正使用。

**改进建议**：直接删除 `pkg/server/gin/middleware/captcha.go` 文件。

---

## 三、配置与部署（Go 特有）

### 3.1 [MEDIUM] 端口 8999 与设计文档 8990 不符

[config/config.yaml:17](file:///E:/DehazeSystem/dehaze-go/config/config.yaml#L17) `system.port: 8999`，但设计文档 [06-部署架构.md](file:///E:/DehazeSystem/dehaze-doc/docs/02-系统架构/06-部署架构.md) 第 37、87、145 行要求 Go 后端端口为 `8990`，Nginx `/go-api/*` → `8990` 的反向代理配置会失效。

config.yaml 第 14-16 行已加注释说明原因：

```yaml
# 端口说明：设计文档（06-部署架构.md）规划生产端口为 8990（Nginx /go-api/* → 8990），
# 但 SDK 集成测试（dehaze-sdk-js test:go）固定连接 8999。
# 当前保持 8999，生产部署时 Nginx upstream 应指向 8999 或修改此端口并同步更新 SDK 配置。
```

**影响**：注释提供了两种解决方案但均未执行，本质问题（端口与设计文档不符）未解决。生产部署时若按设计文档配置 Nginx → 8990 会导致代理失败。

**改进建议**：二选一执行——
1. 修改 config.yaml 为 `port: 8990`，同步更新 `dehaze-sdk-js` 测试配置
2. 修改设计文档 [06-部署架构.md](file:///E:/DehazeSystem/dehaze-doc/docs/02-系统架构/06-部署架构.md) 第 37、87、145 行的 Go 端口为 8999

---

## 四、修复优先级清单

### P2（改进）

| # | 问题 | 文件 |
|---|------|------|
| 1 | Captcha 中间件死代码未删除 | pkg/server/gin/middleware/captcha.go |
| 2 | 端口 8999 vs 设计文档 8990 | config/config.yaml:17, 06-部署架构.md |
