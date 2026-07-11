# dehaze-go 代码评审报告

> 评审范围：`dehaze-go` 全模块 vs `dehaze-doc/docs/03-模块设计` 设计文档
> 评审日期：2026-07-11
> 评审维度：业务契合度 / 架构设计 / 技术规范 / 性能表现 / 代码简化

---

## 概览

| 指标 | 结果 |
|------|------|
| 设计模块数 | 14（核心 6 + 基础 8） |
| Go 代码文件数 | 182（internal）+ 83（pkg） |
| 业务契合度 | 2 / 5 |
| 架构设计 | 3.5 / 5 |
| 技术规范 | 2.5 / 5 |
| 性能表现 | 3 / 5 |
| 代码简化 | 3 / 5 |
| **综合评分** | **2.5 / 5** |

### 总体结论

**不通过**。dehaze-go 目前仅完成了「基础模块」的 RBAC 后台骨架（用户/角色/部门/菜单/字典/认证），以及数据集/算法的「数据模型 + 基础 CRUD」。系统的三大核心业务——**去雾处理、算法选择、效果对比——业务逻辑层完全缺失**，仅有空的数据表定义。文件管理模块的上传/下载/删除物理文件逻辑标注 `TODO` 未实现。算法管理模块的状态机、审核、版本管理均未实现。多个路由缺少权限校验中间件，存在安全漏洞。

核心阻塞问题集中在：
1. 文件上传/下载/删除物理文件未实现（TODO 占位）
2. 去雾处理/算法选择/效果对比三大核心业务完全缺失
3. 算法状态模型与设计严重不符（2 值 vs 6 态状态机）
4. 路由普遍缺少权限校验中间件
5. 算法管理 24 个设计接口仅实现 7 个

---

## 一、业务契合度评审

### [阻塞] 文件上传核心逻辑完全未实现，物理文件从未写入存储

- 位置：`internal/service/file/sys_file.go:29-43`
- 现象：`UploadFile` 方法仅构建 `FileBO` 对象并赋值字段，注释明确写着 `// TODO: 实现文件上传逻辑`。没有调用 MinIO SDK、没有将文件写入任何存储后端。MD5 字段（第 39-40 行）注释为 `// fileBO.MD5 =` 留空。文件上传后物理文件丢失，仅元数据写入数据库。`SaveFile`（第 49 行）根据空字符串 MD5 查询去重，去重逻辑失效。
- 设计依据：需求规格.md 第 3.1.4 节要求"上传文件时计算 MD5 值，如果已存在相同 MD5 的文件，则直接返回已存在记录"；后端实现.md 第 2.1 节要求实现存储策略模式。
- 建议：实现完整的存储抽象层（`StorageService` 接口 + MinIO 实现 + 本地实现），在 `UploadFile` 中使用 `crypto/md5` 流式计算 MD5，调用存储服务完成物理文件写入。

### [阻塞] 文件下载未实现流式传输

- 位置：`internal/service/file/sys_file.go:128-144`
- 现象：`DownloadFile` 注释写着 `// TODO: 实现文件下载逻辑`，仅从数据库查出 `file.Path` 直接返回。`path` 存储的是 `upload/20250119` 这样的相对路径，根本不是可访问的文件系统路径。
- 设计依据：后端实现.md 第 4.1 节要求"流式传输，不加载到内存"，且文件存储在 MinIO。
- 建议：实现 `StorageService.Download` 方法，从 MinIO 获取 `io.ReadCloser` 流，通过 `c.Stream()` 流式写入 HTTP Response。

### [阻塞] 文件删除仅删除数据库记录，物理文件从未被删除

- 位置：`internal/service/file/sys_file.go:81-98`
- 现象：`DeleteFile` 仅调用 `fileRepo.Delete` 删除数据库记录，没有调用任何存储服务的删除方法。
- 设计依据：后端实现.md 第 4.1 节"文件删除流程"明确要求"先删除元数据记录（事务），再删除物理文件"。
- 建议：在 `DeleteFile` 中先删除数据库记录，再调用 `StorageService.Delete(objectName)` 删除物理文件。物理删除失败时记录日志不回滚。

### [阻塞] 去雾处理核心业务逻辑完全缺失

- 位置：`internal/service/task/`（整个目录）、`internal/api/`（无预测 API）、`internal/router/`（无 prediction 路由）
- 现象：需求规格 F-M04-001~F-M04-006 要求实现单张处理、批量处理、参数调节、结果管理、队列管理、错误恢复。Go 端仅有一个 `SysPredLog` 模型定义（`sys_pred_log.go`），没有对应的 Service、Repository、API 层。没有调用 Python 算法服务的 HTTP 客户端，没有异步任务编排，没有进度监控，没有结果保存。`TaskService` 仅实现了导出任务，不支持去雾预测任务。
- 设计依据：去雾处理需求规格.md F-M04-001~006。
- 建议：新增 `internal/service/prediction/` 目录实现 `PredictionService`（含调用 Python `/api/v1/prediction` 的 HTTP 客户端、基于 `(algorithmId, imageMd5)` 的 Redis 缓存、预测日志写入、异步任务发布），新增 `internal/api/sys_prediction.go` 和 `internal/router/prediction.go`。

### [阻塞] 算法选择模块智能推荐功能完全缺失

- 位置：`internal/service/algorithm/sys_algorithm.go`（仅 CRUD）
- 现象：需求规格 F-M03-002 要求实现智能推荐（基于图像特征分析，权重包括雾霾浓度 30%、场景类型 20% 等）。Go 端 `AlgorithmService` 仅实现了基础 CRUD + 树形展示，没有任何图像特征分析、推荐算法、搜索筛选（F-M03-003）、算法收藏（F-M03-005）、算法对比（F-M03-006）。
- 建议：新增推荐服务（调用 Python 端图像特征分析或 Go 端实现特征计算），实现搜索筛选接口，新增算法收藏表及对应 CRUD。

### [阻塞] 效果对比模块评估功能完全缺失

- 位置：`internal/model/sys_eval_log.go`（仅有模型定义）
- 现象：需求规格 F-M05-005 要求实现指标对比模式（PSNR/SSIM/MSE/Entropy 等量化指标评估）。Go 端仅有 `SysEvalLog` 表结构定义，没有对应的 Service、Repository、API 层。没有调用 Python `/api/v1/evaluation` 接口的代码，没有指标计算逻辑。对比 Python 端 `evaluation.py` 已完整实现 PSNR/SSIM/LPIPS/NIQE/Entropy 计算，Go 端完全空白。
- 建议：新增 `internal/service/evaluation/` 目录实现 `EvaluationService`，通过 HTTP 调用 Python 评估接口，管理评估日志，支持多算法对比查询。

### [阻塞] 预测/评估 API 路由完全缺失

- 位置：`internal/app/app.go:154-174`（路由注册）
- 现象：`internal/router/` 目录下没有 prediction、evaluation 相关路由文件。`internal/api/` 目录下没有 `sys_prediction.go`、`sys_evaluation.go`。`app.go` 路由注册中没有任何预测/评估路由。
- 建议：新增对应 api 和 router 文件并在 `app.go` 中注册。

### [阻塞] 算法管理 API 路由大量缺失，覆盖率仅 29%

- 位置：`internal/router/algorithm.go:8-19`
- 现象：设计文档（API 接口.md 2.1-2.5）定义了 24 个接口，Go 端仅注册了 7 个路由。缺失：审核（`PUT /:id/audit`）、版本管理（`POST /:id/version`、`GET /:id/versions`、`POST /:id/rollback`）、导入导出（`POST /_import`、`POST /_export` 等）、监控（`GET /:id/monitor`）、全部预测接口、全部评估接口。此外 `UpdateStatus` 方法存在于 API 层但**路由未注册**该路径，`DELETE /:id` 单个删除路由也缺失。
- 建议：按设计文档逐条补齐路由注册，优先实现审核、版本管理、状态修改路由。

### [阻塞] 算法 Model 层缺失 5 个核心字段

- 位置：`internal/model/sys_algorithm.go:3-19`
- 现象：`SysAlgorithm` 结构体缺失 `version`、`config_json`、`audit_by`、`audit_time`、`audit_remark` 字段。这些字段在数据库设计文档和需求规格中均明确要求。
- 建议：补充对应字段及 GORM tag。

### [阻塞] 算法状态模型严重不匹配 — 仅支持 0/1 两值，设计要求 0-5 六种状态

- 位置：`internal/model/sys_algorithm.go:16`、`internal/model/bo/algorithm.go:13`
- 现象：数据库设计文档定义 status 为 `0:草稿;1:测试中;2:待审核;3:已发布;4:已停用;5:已归档`（6 种状态）。Go 代码中 `Status` 注释为 `1:启用；0:禁用`，BO 层 binding 为 `oneof=0 1`，仅允许 0 和 1。这是旧版本遗留定义。
- 建议：更新为 6 种状态语义，移除 BO 层 `oneof=0 1` 限制，增加状态枚举常量。

### [阻塞] 算法状态流转校验逻辑缺失

- 位置：`internal/service/algorithm/sys_algorithm.go:226-231`
- 现象：`UpdateStatus` 直接调用 `UpdateStatus(ctx, id, status)` 更新状态，完全没有校验算法是否存在、当前状态是否允许转换到目标状态、状态值是否合法。设计文档定义了 8 条状态流转规则，用户可随意绕过审核流程。
- 建议：实现 `validateStatusTransition(currentStatus, targetStatus)` 方法并调用。

### [阻塞] 路由普遍缺少权限校验中间件

- 位置：`internal/router/dataset.go:8-21`、`internal/router/algorithm.go:8-19`、`internal/router/file.go`、`internal/router/item_file.go`、`internal/app/app.go:162-173`
- 现象：所有数据集、算法、文件路由仅注册了 `JWTAuth()` 中间件，未注册任何权限校验中间件。API 接口.md 明确要求写操作需要 `sys:dataset:add/edit/delete`、`sys:algorithm:add/edit/delete` 等权限。当前任何已登录用户都能执行所有写操作，存在安全漏洞。
- 备注：用户/角色/部门/菜单路由已正确使用 `middleware.Permission("sys:role:add")`，但核心业务路由未对齐。
- 建议：为数据集/算法/文件写操作添加权限中间件。

### [阻塞] 数据集删除未实现递归级联删除

- 位置：`internal/service/dataset/sys_dataset.go:652-665`
- 现象：`Delete` 方法直接调用 `datasetRepo.Delete`（逻辑删除），没有递归删除子数据集、没有删除关联数据项和文件。需求规格 F-M06-002 明确要求"删除时递归删除所有子数据集""删除时同时删除关联的所有图片数据"。注意 `BatchDeleteDatasets` 实现了完整级联删除，但单个删除接口走的是 `DatasetService.Delete`。
- 建议：`DeleteDataset` API 应调用 `operationService.BatchDeleteDatasets`。

### [阻塞] 数据集列表树形结构不完整

- 位置：`internal/api/sys_dataset.go:37-52`
- 现象：`GetDatasetList` 调用 `FindRootPage` 仅查询根节点，再加载一层子节点，**不是完整树形结构**，而是"根+一级子节点"的浅层展开。3 级以上层级树形结构不完整。后端实现.md 第 4.1.1 节要求"一次性查询所有数据集，内存构建父子关系（BFS 算法）"。
- 建议：提供独立的 `GET /datasets/tree` 接口返回完整树，或在 `GetDatasetList` 中返回完整树。

### [警告] 数据项分页查询缺少关键字搜索和相关度排序

- 位置：`internal/api/sys_dataset_item.go:62-95`、`internal/service/dataset/sys_dataset_item.go:130-207`
- 现象：需求规格 F-M06-007 要求支持按文件名、描述、场景类型搜索并按相关度降序排序（后端实现.md 定义了相关度计算规则）。当前仅支持 `datasetId` 和 `sceneType` 两个查询参数，不支持关键字搜索，无相关度排序。
- 建议：扩展 `DatasetItemQuery` 增加关键字字段，Repository 层实现带相关度计算的搜索。

### [警告] 数据项分页查询缺少按雾霾程度筛选

- 位置：`internal/api/sys_dataset_item.go:62-95`
- 现象：需求规格 F-M06-006 要求支持按雾霾程度（light/medium/heavy/未标注）筛选，当前无 `hazeLevel` 参数。
- 建议：增加 `hazeLevel` 查询参数，通过 JOIN `sys_item_file` 表筛选。

### [警告] 缺少数据集导出/下载任务接口

- 位置：`internal/router/dataset.go`（无导出路由）
- 现象：API 接口.md 第 2.4 节定义了统一任务接口（`/api/v1/tasks`），支持 `dataset_export`、`item_download`、`batch_download`。Go 代码中完全没有实现任务路由和 Controller，`DatasetExportStrategy` 等策略类未实现。
- 建议：实现任务管理模块的路由和策略注册。

### [警告] 图像输入模块缺少历史记录管理功能

- 位置：整个 `internal/` 目录
- 现象：设计文档图像输入后端实现.md 第 4 节描述了历史记录管理（`sys_input_history` 表的 CRUD、配额管理、同步机制），API 接口.md 第 2.3 节定义了 8 个历史记录接口。Go 代码中完全没有 `sys_input_history` 相关代码。
- 建议：新增 `model/sys_input_history.go` 及对应 service/repository/api/router。

### [警告] WPX 格式转换功能未实现

- 位置：`internal/api/sys_file.go:72-78`
- 现象：`UploadFile` 中 modelId 参数处理逻辑为 `// TODO: 实现获取WPX文件的逻辑`。`model/sys_wpx_file.go` 已定义但从未被引用。
- 建议：实现 WpxService，在 modelId 存在时调用格式转换。

### [警告] 缺少文件大小限制校验

- 位置：`internal/api/sys_file.go:34-81`
- 现象：`UploadFile` API 没有任何文件大小校验。需求规格要求"默认最大文件大小 100MB"。
- 建议：添加 `file.Size > maxSize` 检查，配置 Gin 的 `MaxMultipartMemory`。

### [警告] MD5 校验接口返回值类型与文档不符

- 位置：`internal/api/sys_file.go:120-131`、`internal/service/file/sys_file.go:75-79`
- 现象：设计文档要求返回 `Boolean`（true=已存在），Java 实现返回完整 `SysFile` 对象。Go 实现返回 `bool`，前端获取不到文件 ID 等信息，无法实现秒传。同时缺少 MD5 格式校验（32 位十六进制）。
- 建议：返回文件完整信息而非布尔值，增加 MD5 格式校验。

### [警告] SysPredLog / SysEvalLog 模型字段不完整

- 位置：`internal/model/sys_pred_log.go:6-20`、`internal/model/sys_eval_log.go:6-21`
- 现象：`SysPredLog` 缺少 `Status`（任务状态）、`Params`（处理参数 JSON）、`Progress`（进度）、`ErrorMessage`（错误信息）。`SysEvalLog` 缺少 `Status`、`Progress`、`ErrorMessage`、`AlgorithmName`。
- 建议：补充字段以支持异步任务状态跟踪。

### [警告] SysTask 任务类型枚举缺少去雾/预测/评估类型

- 位置：`internal/model/sys_task.go:19-25`
- 现象：`TaskType` 枚举仅有 `export/import/thumbnail/compression/cleanup`，没有 `prediction` 和 `evaluation`。
- 建议：增加对应枚举值。

### [警告] VIP 权限控制逻辑完全缺失

- 位置：Go 全项目
- 现象：去雾处理需求规格第 4 节要求 VIP 权益控制（每月去雾次数限制：普通 20 次/VIP1 100 次/VIP2 500 次/SVIP 3000 次），效果对比需求规格第 5 节也要求多算法对比的 VIP 限制。Go 端无任何相关实现。
- 建议：新增 VIP 权益服务，对接用户角色进行次数限制。

### [警告] 验证码类型与文档不匹配

- 位置：`internal/service/auth/auth_service.go:250-254`
- 现象：代码使用 `base64Captcha.NewDriverDigit` 生成数字验证码，设计文档定义了 LINE/CIRCLE/SHEAR/GIF 四种类型，默认 CIRCLE。
- 建议：根据配置项 `captcha.type` 动态选择验证码驱动。

### [警告] RefreshToken API 从 Header 获取而非请求体

- 位置：`internal/api/auth.go:127-128`
- 现象：`RefreshToken` 从 `security.GetToken(c)` 获取当前 Token，而非从请求体获取 `refreshToken` 参数。设计文档第 3.4.2 节要求输入参数为 `refreshToken`。
- 建议：从请求体 JSON 解析 `refreshToken` 字段。

---

## 二、架构设计评审

### [亮点] 显式依赖注入，启动链路清晰

- 位置：`internal/app/app.go:91-174`
- 现象：采用构造函数注入（显式 wiring），按 repo → service → api → router 顺序装配，避免了运行时 DI 容器的复杂性。初始化顺序明确：配置 → 日志 → 数据库 → 缓存 → HTTP Server → validator → 业务 wiring。
- 评价：设计优秀，符合 Go 社区偏好。

### [亮点] 优雅停机按依赖反序关闭资源

- 位置：`internal/app/app.go:197-242`
- 现象：`shutdown` 按依赖反序关闭：HTTP Server → TaskExecutor（RabbitMQ）→ Cache → Database → Logger（最后 flush）。收集所有错误统一返回。
- 评价：资源释放顺序正确，错误处理完善。

### [亮点] 多级缓存架构设计完整

- 位置：`pkg/cache/manager.go`、`pkg/cache/multilevel/cache.go`、`pkg/cache/protection/`
- 现象：实现了 L1（本地）+ L2（Redis）多级缓存，集成布隆过滤器（防穿透）、SingleFlight（防击穿）、熔断器（防雪崩）、空值缓存、Pub/Sub 失效广播。防护组件通过接口抽象可灵活组合。
- 评价：缓存防护体系完善。

### [亮点] 权限校验中间件设计合理

- 位置：`pkg/security/permission.go:137-155`、`internal/router/user.go:37-41`
- 现象：通过 `middleware.Permission("sys:role:add")` 进行接口级权限校验，先检查 JWT claims 中的 authorities，支持 Casbin 可选集成，使用 `sync.Map` 本地缓存 + Redis Pub/Sub 多实例失效广播（5 分钟 TTL）。
- 评价：与 Java 端 `@PreAuthorize` 效果一致，性能优化到位。

### [亮点] 双 Token 机制已实现（超前于设计文档）

- 位置：`pkg/security/claims.go:153-186`、`internal/service/auth/auth_service.go:79-80`
- 现象：设计文档标注"当前版本未实现双 Token 机制"，但 Go 代码已完整实现 `LoginTokenWithRefresh`，生成 accessToken + refreshToken 对。还实现了防暴力破解（IP + 用户名双重维度登录失败锁定）。
- 评价：安全增强，建议同步更新文档。

### [警告] 核心业务路由普遍缺少权限中间件（架构一致性缺失）

- 位置：见业务层阻塞项
- 现象：基础模块（user/role/dept/menu/dict）已正确使用 `middleware.Permission()`，但核心业务模块（dataset/algorithm/file/item_file）全部缺失。同一项目内权限校验策略不一致，属于架构层面的遗漏。
- 建议：统一所有写操作路由的权限中间件策略。

### [警告] 部门管理和任务管理缺少 API 层和路由层

- 位置：`internal/api/`（无 sys_dept.go 实际路由注册确认）、`internal/router/`（无 dept.go、task.go）
- 现象：dept 有 service + repository + model 但需确认 api 层完整性；task 有 service + repository + model 但**完全没有 api 和 router**。任务管理作为统一任务接口的承载方，缺少对外暴露的 API 意味着任务功能无法被前端调用。
- 建议：补齐 task 的 api 和 router，注册到 `app.go`。

### [警告] CustomClaims 缺少 GetDeptID 和 GetDataScope 方法

- 位置：`pkg/security/claims.go:17-23`
- 现象：`DataScopePlugin` 尝试通过接口获取 `GetDataScope()`、`GetDeptID()`，但 `CustomClaims` 只实现了 `GetUserID()`，未实现另外两个方法，导致数据权限插件回退到反射方式获取，性能和可靠性受影响。
- 建议：为 CustomClaims 补充 `GetDataScope() int8` 和 `GetDeptID() int64` 方法。

### [警告] 配置热更新 SystemEvents 遇首个错误即中止

- 位置：`config/system_events.go:24-34`
- 现象：`TriggerReload` 遍历所有 handler，遇到第一个错误即返回，后续 handler 不会执行，导致部分组件未收到重载通知。
- 建议：收集所有错误后统一返回，或记录日志后继续执行。

### [通过] 分层架构清晰

- 位置：`internal/` 整体结构
- 现象：api → service → repository → model 四层分层清晰，service 层按业务域拆分子包（algorithm/auth/dataset/dept/dict/file/menu/role/task/user），repository 同构。model 层细分 bo/dto/vo/query/enum/read。
- 评价：符合 DDD 风格的领域分层。

### [通过] 存储抽象设计合理（接口已定义但未实现）

- 位置：文件管理模块接口设计
- 评价：存储策略模式的设计方向正确，但实现缺失。

---

## 三、技术规范评审

### [阻塞] Redis 分布式锁实现不安全

- 位置：`pkg/cache/redis/impl.go:130-148`
- 现象：`Lock` 使用 `SetNX` 加锁，但 `Unlock` 直接 `Del` 删除 key，没有校验锁的持有者。任何客户端都可以释放他人持有的锁，存在并发安全问题。
- 建议：加锁时 value 写入唯一标识（UUID），解锁时使用 Lua 脚本校验 value 后再删除。

### [阻塞] 密码通过 Query 参数传递

- 位置：`internal/api/sys_user.go:244`
- 现象：`UpdatePassword` 通过 `c.Query("password")` 从 URL 查询参数获取新密码，密码会出现在 URL 中，可能被日志、浏览器历史、代理服务器记录。
- 建议：改为通过请求体（Body）传递，使用 POST/PATCH + JSON body。

### [警告] Token 黑名单 TTL 使用固定配置而非 Token 剩余有效期

- 位置：`internal/service/auth/auth_service.go:308-310`
- 现象：`AddTokenToBlacklist` 使用 `cfg.JWT.TTL` 作为黑名单 TTL，但设计文档要求"计算剩余有效期：`ttl = exp - currentTime`"。黑名单会过度占用 Redis 内存。
- 建议：从 Token claims 解析 `exp`，计算实际剩余有效期。

### [警告] 验证码缓存过期时间不一致

- 位置：`pkg/security/captcha.go:32`
- 现象：代码中 `Expiration: time.Second * 180`（3 分钟），设计文档规定 120 秒（2 分钟）。
- 建议：统一为 120 秒或从配置读取。

### [警告] 验证码校验未区分"过期"和"错误"

- 位置：`internal/service/auth/auth_service.go:55-58`
- 现象：`VerifyCaptcha` 返回 false 时统一返回"验证码错误"，设计文档区分了验证码错误（A0230）和验证码已过期（A0231）。
- 建议：修改 `VerifyCaptcha` 返回错误类型，分别返回 A0231 和 A0230。

### [警告] CaptchaStore.Get 方法存在 Bug

- 位置：`pkg/security/captcha.go:60-61`
- 现象：`Get(key, clear)` 直接使用 `key` 查询 Redis，但 `Set` 存储时使用 `rs.PreKey + id`。key 前缀不一致导致校验失效。
- 建议：统一 key 拼接逻辑。

### [警告] 文件服务使用 context.Background() 而非请求上下文

- 位置：`internal/service/file/sys_file.go:48, 76, 83, 102, 133`
- 现象：多个方法使用 `context.Background()` 而非从 API 层传入的 `c.Request.Context()`。请求取消时 Service 层无法感知，长耗时操作（如文件下载）无法被中断。
- 建议：所有 Service 方法接收 `context.Context` 参数并透传。

### [警告] 本地缓存 IncrBy 非原子操作

- 位置：`pkg/cache/local/impl.go:119-150`
- 现象：`IncrBy` 先 `Get` 再 `Set`，中间没有加锁，并发调用会导致计数丢失。
- 建议：使用 `sync.Map` 的 `LoadOrStore` + CAS 循环，或对每个 key 使用独立锁。

### [警告] 本地缓存 Lock 不支持过期

- 位置：`pkg/cache/local/impl.go:193-209`
- 现象：`Lock` 接受 `expiration` 参数但完全忽略，锁永远不会自动过期。持有锁的进程崩溃后锁将永久存在。
- 建议：使用 `time.AfterFunc` 在过期后自动清理锁。

### [警告] 本地缓存 TTL 管理存在内存泄漏

- 位置：`pkg/cache/local/impl.go:40-50, 162-185`
- 现象：`Set` 在 `expiration > 0` 时向 `ttlMap` 存储 TTL，但仅当 key 不存在时才存储。对同一 key 先 Set 短 TTL 再 Set 长 TTL，TTL 不会更新。`ttlMap` 中过期 key 不会被主动清理。
- 建议：每次 `Set` 都更新 `ttlMap`；增加后台 goroutine 定期扫描过期 key。

### [警告] 本地缓存 Hash/Set 数据无 TTL 和大小限制

- 位置：`pkg/cache/local/impl.go:268-404`
- 现象：`hashData` 和 `setData` 使用 `sync.Map` 存储，没有过期机制也没有大小限制，长期运行会无限增长。
- 建议：增加 TTL 管理和最大条目数限制。

### [警告] Redis 连接池配置硬编码且不合理

- 位置：`pkg/cache/redis/init.go:24-29`
- 现象：`PoolSize: 10` 硬编码，未从配置读取。未设置 `MinIdleConns`、`PoolTimeout`、`IdleTimeout`。配置中 `Timeout` 字段也未被使用。
- 建议：将连接池参数暴露到配置项，至少设置 `PoolTimeout` 避免连接获取无限等待。

### [警告] multilevel MGet 用空字符串区分 miss 与未设置

- 位置：`pkg/cache/multilevel/cache.go:226-233`
- 现象：MGet 从 L1 获取结果时用 `val != ""` 判断是否命中。如果缓存值本身是空字符串，会被误判为 miss。
- 建议：使用 `(string, bool)` 返回值区分 miss 和空值。

### [警告] CacheManager.Init 使用 sync.Once 导致初始化失败后不可重试

- 位置：`pkg/cache/manager.go:47-89`
- 现象：`once.Do` 保证只执行一次。如果 Redis 初始化失败，`cacheManager` 处于部分初始化状态，后续调用 `Init()` 不会重试。
- 建议：参照 database 组件的 `initState` 模式，允许失败后重试。

### [通过] 密码加密使用 BCrypt

- 位置：`internal/service/user/user_service.go:62, 208, 302, 326, 463`
- 评价：符合安全要求。

### [通过] JWT Claims 格式符合项目统一要求

- 位置：`pkg/security/claims.go:17-23`
- 评价：包含 `jti/sub/userId/authorities/deptId/dataScope`，符合三端互认要求。

---

## 四、性能评审

### [警告] 数据集树形查询可能存在 N+1 问题

- 位置：`internal/service/dataset/sys_dataset.go`（树构建逻辑）
- 现象：`GetDatasetList` 先查根节点分页，再 `FindByParentIDs` 查子节点。如果层级较深或子节点较多，可能产生多次数据库查询。
- 建议：一次性查询所有数据集，内存构建树（BFS），与后端实现.md 一致。

### [警告] SingleFlight context 取消后 goroutine 泄漏

- 位置：`pkg/cache/protection/singleflight.go:62-78`
- 现象：当 `ctx.Done()` 时 goroutine 可能泄漏。
- 建议：确保 context 取消时清理 goroutine。

### [优化] 算法列表查询未利用缓存

- 位置：`internal/service/algorithm/sys_algorithm.go`
- 现象：算法数据变更频率低，但每次查询都走数据库。对比菜单/角色/部门已使用 Redis 缓存。
- 建议：为算法列表和选项数据增加缓存层。

### [优化] 数据集分页查询排序字段未优化

- 位置：`internal/repository/dataset/dataset_item_repository.go:88-104`
- 现象：仅按 `id DESC` 排序，未针对常见查询场景（如按创建时间、按名称）建立复合索引。
- 建议：根据查询模式建立合适索引。

### [通过] 权限检查使用本地缓存 + Redis 双层缓存

- 位置：`pkg/security/permission.go:36-42, 308-344`
- 评价：`sync.Map` 本地缓存 5 分钟 TTL + Redis Pub/Sub 多实例失效广播，性能优化到位。

### [通过] 配置读写锁保护

- 位置：`pkg/config/config.go:21-39`
- 评价：`sync.RWMutex` 保护全局配置指针，并发安全。

### [通过] 异步任务执行器基于 RabbitMQ

- 位置：`internal/service/task/`、`pkg/mq/`
- 评价：任务异步化设计方向正确（虽然去雾预测任务未接入）。

---

## 五、代码简化评审

### [警告] 空文件未清理

- 位置：`pkg/config/loader.go`、`pkg/config/watcher.go`
- 现象：两个文件仅有 `package config` 声明，无任何代码。
- 建议：删除空文件，或补充实现。

### [警告] 配置 default tag 误导

- 位置：`pkg/config/options/db.go:13-29`
- 现象：DB 配置结构体使用 `default` tag 标注默认值，但 viper 的 `Unmarshal` 不会自动应用 `default` tag。实际默认值依赖 `Config.Validate()` 中的硬编码补全。
- 建议：移除无用的 `default` tag 避免误导，或引入默认值填充逻辑统一处理。

### [优化] getConfigName 未处理未知 gin.Mode

- 位置：`pkg/config/viper.go:16-27`
- 现象：当 `gin.Mode()` 不匹配 Debug/Release/Test 时，返回零值空字符串，可能导致配置文件查找失败。
- 建议：在 default 分支中 panic 或返回 error。

### [优化] 文件服务重复的 context.Background() 调用

- 位置：`internal/service/file/sys_file.go:48, 76, 83, 102, 133`
- 现象：每个方法都单独创建 `context.Background()`，既不符合规范又重复。
- 建议：统一接收上层 context 参数。

### [优化] rand.Int63n 线程安全依赖 Go 版本

- 位置：`pkg/cache/multilevel/cache.go:695`
- 现象：`rand.Int63n` 在 Go 1.20 之前不是并发安全的。需确认项目 Go 版本。
- 建议：确认 Go 版本 >= 1.20，或使用 `rand.New(rand.NewSource(...))` 配合锁。

### [通过] 命名规范一致

- 位置：整体代码
- 评价：Go 标准命名风格，包名简短，类型名驼峰，接口前缀 I 清晰。

### [通过] 错误处理使用统一 BizError 封装

- 位置：`pkg/common/`、各 service 层
- 评价：业务错误码 + 中文消息 + 原始错误包装，统一规范。

---

## 改进建议（按优先级排序）

### 高优先级（阻塞，必须修复）

1. **实现文件上传/下载/删除的物理文件处理** — 当前 TODO 占位导致文件功能完全不可用
2. **实现去雾处理/算法选择/效果对比三大核心业务** — 系统核心价值所在，当前完全缺失
3. **补齐算法管理 API 路由**（审核/版本/状态修改/导入导出/监控）— 覆盖率仅 29%
4. **修正算法状态模型**（0/1 → 0-5 六态）并实现状态流转校验
5. **为所有写操作路由添加权限校验中间件** — 当前存在安全漏洞
6. **修正 Redis 分布式锁实现**（Unlock 需校验持有者）
7. **密码改为请求体传递**（当前通过 URL Query 泄漏）
8. **补充 SysAlgorithm 缺失的 5 个字段**
9. **数据集删除走级联删除逻辑**
10. **补齐预测/评估 API 路由注册**

### 中优先级（警告，建议修复）

11. 文件服务统一使用请求 context 而非 context.Background()
12. Token 黑名单 TTL 改为 Token 剩余有效期
13. 验证码类型/过期时间/错误码与文档对齐
14. 修复 CaptchaStore.Get 的 key 前缀 Bug
15. CustomClaims 补充 GetDeptID/GetDataScope 方法
16. 数据集列表改为完整树形结构
17. 数据项查询增加关键字搜索和相关度排序
18. 实现图像输入历史记录管理
19. 补齐任务管理 API 层和路由层
20. 修复本地缓存 IncrBy/Lock/TTL 多个并发与内存问题
21. Redis 连接池参数可配置化
22. multilevel MGet 区分 miss 与空值

### 低优先级（优化）

23. 算法列表增加缓存层
24. 清理空文件和无用 default tag
25. getConfigName 处理未知 gin.Mode
26. 确认 rand 线程安全性

---

## 亮点

1. **显式依赖注入 + 优雅停机**：`app.go` 的构造函数注入和依赖反序关闭设计优秀
2. **多级缓存架构**：L1+L2 + 布隆过滤器 + SingleFlight + 熔断器 + Pub/Sub 失效广播，防护体系完善
3. **双 Token 机制 + 防暴力破解**：超前于设计文档实现，安全增强
4. **权限检查双层缓存**：本地 `sync.Map` + Redis Pub/Sub，性能优化到位
5. **配置热更新**：fsnotify 监听 + 环境变量展开 + validator 校验 + 事件广播
6. **分层架构清晰**：api → service → repository → model 四层 + model 内部 bo/dto/vo/query 细分
7. **统一错误处理**：BizError 封装，错误码 + 消息 + 原始错误包装

---

## 评审结论

**结论：不通过**。

dehaze-go 目前处于「基础后台骨架已完成、核心业务未实现」的阶段。基础模块（用户/角色/部门/菜单/字典/认证）的 RBAC 体系和 pkg 公共组件（缓存/配置/安全/服务器）设计质量较高，有多个亮点。但系统的核心业务价值——去雾处理、算法选择、效果对比——**业务逻辑层完全缺失**；文件管理模块的物理文件处理标注 TODO 未实现；算法管理模块的状态机、审核、版本管理未实现；多个路由缺少权限校验中间件。

**建议路径**：
1. 第一阶段：修复所有安全阻塞问题（权限中间件、密码传递、分布式锁）
2. 第二阶段：实现文件上传/下载/删除物理文件处理（存储抽象层 + MinIO 接入）
3. 第三阶段：补齐算法管理完整功能（状态机、审核、版本管理、路由补全）
4. 第四阶段：实现三大核心业务（去雾处理、算法选择、效果对比），对接 Python 算法服务
5. 第五阶段：修复缓存组件并发问题，补齐数据集树形/搜索/导出功能

在核心业务实现并修复所有阻塞问题前，不建议将 dehaze-go 作为生产后端使用。
