# Java 后端架构改造计划

> 本文档记录 dehaze-java 代码架构层面的实际问题与改造方向，供后续重构参考。文档失真问题已在 [01-Java架构文档.md](../04-项目实现/后端/01-Java架构文档.md) 修复中处理，本文聚焦**代码与架构本身**的债务。
>
> 三端对照结论：dehaze-go 与 dehaze-java 在 MQ 拓扑上已对齐（两端 MQ 均为 `task.export` + `feedback.low_rating` + 3 级重试 + DLX）；定时任务两端接近对齐（Java 15 个 XxlJob handler、Go 14 个，Java 多 `processDelayedPush`）；dehaze-python 业务能力落后（仅 5 个 XxlJob，缺订单/会员/消息/公告等业务 Job）。推荐模块伪特征是 **Java + Go 共有** 债务，Python 端已实现真实图像分析服务但未被两端调用（详见 §1.3）。

## 一、问题清单

### 1.1 数据模型分层混乱【P1】

**现状**：`model/` 包下 7 个子包，但实际使用严重失衡：
- 主力层：`entity/`(47)、`form/`(65)、`vo/`(92)、`query/`(28)
- 按需层：`bo/`(6)、`event/`(2)
- **混乱层**：`dto/`(10) 包内混入 `LoginForm`、`RegisterForm` 等 Form 后缀对象，与 `form/` 包命名规则冲突，新成员无法判断入参应放 `dto/` 还是 `form/`

**影响**：模型归属无规则可循，`dto/` 包成为"既非纯 DTO 又非纯 Form"的杂物间，增加维护与理解成本。

**改造方向**：
1. 将 `dto/LoginForm`、`dto/RegisterForm` 迁移至 `form/` 包，统一 Form 后缀对象的归属
2. `dto/` 包仅保留真正的跨层数据传输对象（`LoginResult`、`CaptchaResult`、`ApiKeyResult`、`FileInfo`、`ImageFileInfo`、`DatasetStatistics`、`UserAuthInfo`、`ChatMessage`）
3. 评估 `bo/`（仅 6 文件）是否有独立存在价值：若仅 `UserBO`、`FileBO` 等少量跨服务聚合对象，可合并入 `dto/` 或直接在 Service 内部使用 Entity+局部字段，消除近空置的分层

### 1.2 去雾异步状态存储分叉【P1】

**现状**：去雾处理状态存于 `sys_input_history.status`，而非统一任务表的 `sys_task.status`。`SysInputHistoryService` 既是历史 CRUD 又承载状态机职责。

**影响**：
- 状态查询、清理、统计需走两套路径（`SysInputHistoryService` vs `TaskService`）
- 与"统一任务管理框架"的架构承诺背离，任务管理模块无法纳管去雾任务
- `SysInputHistoryService.syncHistory()` 是占位接口（"后续实现云端同步"），表明该服务职责边界本就模糊

**改造方向**：
- **方案 A（推荐，与 Go 端对齐）**：去雾任务状态归 `sys_pred_log`，`sys_input_history` 退化为纯历史视图（关联 `pred_log_id`），状态查询走统一路径。Go 端已采用此设计（预测状态存于 `sys_pred_log.status`，`input_history_service.go` 仅做 CRUD 无状态机），Java 应向 Go 靠拢。
- **方案 B**：若历史视图确需独立，明确 `SysInputHistoryService` 仅做历史查询，状态机完全委托 `TaskService`，移除 `syncHistory()` 占位接口
- ~~改造需同步 Go 端（Go 端 input_history_service.go 存在同样分叉）~~ —— **此前的描述失实**：Go 端 `input_history_service.go` 仅做输入历史 CRUD，不承载预测状态，不存在此分叉。Java 端为独立债务，无需 Go 协同。

### 1.3 推荐模块伪特征【P1，跨端】

**现状**：`RecommendationServiceImpl.analyze`（`RecommendationServiceImpl.java:70-73`）基于图片 URL 的 MD5 哈希生成确定性"伪特征"，代码注释明确"实际生产环境应调用 Python 算法服务提取真实特征"。Go 端 `RecommendationService.Analyze` 为同款实现。

Python 端已实现真实图像分析服务（PyTorch 场景分类 + OpenCV 暗通道/边缘/直方图，见 [Python 架构文档 §3.8](../04-项目实现/后端/03-Python算法服务架构文档.md)），但 Java 与 Go 均**未调用**——Python 端图像分析服务目前是未被调用的死基础设施。

**影响**：
- 推荐结果基于虚假特征，场景匹配（urban/landscape/building 等）纯随机，推荐无业务价值，与推荐管理模块需求规格（F-REC-001 图像特征分析）背离
- 三端业务"对等"承诺失效：Python 端真实分析 vs Java/Go 端 mock，同一图片在三端得到不同推荐质量

**改造方向**：
1. Java 端新增调用 Python 图像特征分析接口的客户端方法（复用现有算法服务 HTTP 客户端的重试/熔断/幂等机制）
2. `RecommendationServiceImpl.analyze` 改为调用 Python 服务，移除 MD5 伪特征逻辑
3. Python 服务不可用时降级策略：返回错误（推荐模块不可用）而非伪特征，避免误导用户
4. **同步 Go 端**：Go `RecommendationService.Analyze` 需同样改造（详见 [Go 改造计划 §二](./Go后端架构改造计划.md)）
5. 修正 Python 架构文档 §3.8 关于"Java/Go 已调用"的失实描述，待两端接入后再如实记录

**验收标准**：
- Java/Go 端 `analyze` 返回真实图像特征，与 Python 端直接调用特征分析服务结果一致
- Python 服务不可用时返回明确错误，不返回伪特征
- 推荐管理模块测试用例覆盖真实特征输入

## 二、三端一致性改造

### 2.1 Python 端业务 Job 补齐【P1，跨端】

**现状**：Java 15 个、Go 14 个 XxlJob handler，Python 仅 5 个。Python 端缺失：
- 订单：expireOrders、completeExpiredOrders、retryFailedRefunds、autoRenew、expireUserCoupons
- 会员：resetMonthlyQuota、processExpiredMembers、sendExpireReminders
- 消息：cleanupExpiredMessages、refreshUnreadCountCache、processDelayedPush
- 公告：sendScheduledAnnouncements

**影响**：三端声称业务对等，但定时任务层面 Python 端无法独立承担订单/会员/消息等业务运维，依赖 Java 或 Go 端调度。

**改造方向**：Python 端补齐业务 Job，或明确分工（如 Python 仅承担算法相关 Job，业务 Job 由 Java/Go 承担）并更新三端架构文档对照表。**此项主体在 dehaze-python，Java 端无需代码改造，但需在三端对照表中如实标注差异。**

## 三、改造优先级与建议时序

| 优先级 | 改造项 | 主体端 | 建议时序 |
|--------|--------|--------|----------|
| P1 | 1.1 数据模型分层混乱（dto/form 归属） | Java | 近期，纯重构无外部影响 |
| P1 | 1.2 去雾状态存储分叉 | Java | 近期，Java 独立改造（向 Go 设计靠拢） |
| P1 | 1.3 推荐模块伪特征接入 Python | Java + Go | 近期，需两端协同 + Python 接口确认 |
| P1 | 2.1 Python 业务 Job 补齐 | Python | 近期，Java 端仅更新对照表 |

## 四、不纳入本计划的事项

- **文档失真修复**：已在 [01-Java架构文档.md](../04-项目实现/后端/01-Java架构文档.md) 上一轮修复中完成，不重复。Go 架构文档的服务虚构、is_favorite 冗余描述等也已同步修复，无需再列为改造项。
- **业务功能扩展**（多任务类型、OpenAPI/SDK）：见 [产品拓展升级规划](./产品拓展升级规划.md)
- **算法模型加载治理**：见 [近期改造计划总览 §5](./近期改造计划总览.md)
- **Go/Python 端代码架构问题**：见 [Go 改造计划](./Go后端架构改造计划.md) / [Python 改造计划](./Python后端改造计划.md)，本计划聚焦 Java，其他端问题仅在影响三端一致性时提及

## 五、经评审剔除的伪债务/低价值项

以下项经代码核实后确认不构成债务或已实现，记录于此供参考，不纳入改造：

| 原项 | 剔除原因 |
|------|---------|
| ~~WebSocket 跨实例广播缺失~~（原 §1.6） | **已实现**：`config/WebSocketMessageRelay.java` 通过 Redis Pub/Sub 跨实例广播，类注释明确"对齐 Python 端方案"，非债务 |
| ~~评估结果永久缓存无淘汰~~（原 §1.3） | **无法证实**：Python 应用代码 grep 不到 `{algorithmId}:{predMd5}:{refMd5}` 缓存实现，疑似基于文档臆测；且改造主体本就在 Python，不应列入 Java 计划 |
| ~~model/service 分包不一致~~（原 §1.4） | **低价值**：model 按类型分包是 Java/Spring 生态惯例，改为按业务域分包违背社区主流实践，收益不抵风险 |
| ~~XxlJobConfig 半集成~~（原 §1.5） | **低价值**：属环境配置问题非代码架构债务，`@ConditionalOnProperty` 扫描开销可忽略 |
| ~~三端文档虚构治理~~（原 §2.2） | **已完成**：Java 与 Go 架构文档均已在上一轮修复，无需再列为改造项 |
| ~~收藏 is_favorite 澄清~~（原 §2.3） | **已完成**：Go 文档已修复，无残留 |
