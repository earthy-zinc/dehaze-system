# Java 后端架构改造计划

> 本文档记录 dehaze-java 代码架构层面的实际问题与改造方向，供后续重构参考。文档失真问题已在 [01-Java架构文档.md](../04-项目实现/后端/01-Java架构文档.md) 修复中处理，本文聚焦**代码与架构本身**的债务。
>
> 三端对照结论：dehaze-go 与 dehaze-java 在 MQ 拓扑上已对齐（两端 MQ 均为 `task.export` + `feedback.low_rating` + 3 级重试 + DLX）；定时任务两端接近对齐（Java 15 个 XxlJob handler、Go 14 个，Java 多 `processDelayedPush`）；dehaze-python 业务 Job 已补齐（17 个，含 Python 专属的模型健康检查/孤儿文件/临时文件清理）。§1.2 的 `SysInputHistoryService` 死代码是 **Java + Go 共有** 债务（两端均实现了 syncHistory 假接口和 updateHistory 空壳），需跨端同步清理。

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

### 1.2 SysInputHistoryService 死代码清理【P2，跨端】

**现状**：`SysInputHistoryService` 是图像输入历史记录模块，跨三端 + SDK 完整实现（Java/Go 后端、dehaze-sdk-js 封装、uniapp/react-native 前端组件接入）。经代码核实，该服务存在三处死代码/空壳实现：

1. **`syncHistory()` 假实现**：方法体仅将当前用户的 `syncStatus` 字段从 0 更新为 1，无任何云端同步逻辑。`syncStatus` 字段无任何消费方，纯属占位。
2. **`updateHistory()` 空壳**：方法注释称"具体更新字段由前端决定"，实际仅调用 `updateById(history)` 且未对 `HistoryUpdateForm` 的任何字段做赋值——即啥也不更新。
3. **`status` 字段语义模糊**：`SysInputHistory.status`（1=成功/2=失败/3=处理中）由 `createHistory` 时前端传入，**创建后从不更新**。它是历史记录的业务属性（创建时确定），不是异步任务状态机，但注释和命名易被误认为会随处理进度推进。

**补充说明**：此前版本曾将本项误判为"去雾异步状态存储分叉"（误以为 `sys_input_history.status` 与 `sys_task.status` 存在状态机分叉）。经核实，预测主流程（`prediction/` 拦截器链）完全不调用 `SysInputHistoryService`，两表是独立功能，不存在分叉关系。

**影响**：
- `syncHistory` + `syncStatus` 是纯粹的死代码，增加维护噪声和表字段冗余
- `updateHistory` 空壳接口暴露给前端但无实际效果，若前端调用会造成"更新成功但数据未变"的误导
- `status` 字段语义不清，易诱使后续维护者误加状态更新逻辑

**改造方向**：
1. **移除 `syncHistory()` + `syncStatus` 字段**：无云端同步需求，删除 Java/Go 的接口与实现、SDK 的 `sync()` 方法、Controller 的 `/sync` 端点、前端测试用例
2. **移除或实现 `updateHistory()`**：当前无真实更新需求，建议直接移除接口（含 `HistoryUpdateForm`、Controller 的 PUT 端点、SDK 的 `update()` 方法）；若后续有更新需求再按需新增
3. **`status` 字段注释澄清**：在 `SysInputHistory` Entity 与 `InputHistoryVO` 注释中明确"创建时确定，不随处理进度更新"，消除状态机误解
4. **数据库迁移**：`sys_input_history` 表删除 `sync_status` 列（若存在 `update_by` 等仅服务于 syncHistory 的字段一并清理）
5. **同步范围**：Java、Go、dehaze-sdk-js（含测试）、dehaze-uniapp、dehaze-react-native、dehaze-doc 模块设计文档

**验收标准**：
- 三端后端无 `syncHistory`/`syncStatus`/`updateHistory`/`HistoryUpdateForm` 残留
- SDK 无 `sync()`/`update()` 方法及对应测试
- `sys_input_history` 表无 `sync_status` 列
- `status` 字段注释明确为"创建时确定的业务属性"

## 二、改造优先级与建议时序

| 优先级 | 改造项 | 主体端 | 建议时序 |
|--------|--------|--------|----------|
| P1 | 1.1 数据模型分层混乱（dto/form 归属、bo 层去留） | Java | 近期，纯重构无外部影响 |
| P2 | 1.2 SysInputHistoryService 死代码清理 | Java + Go + SDK + 前端 | 近期，跨端同步删除 syncHistory/updateHistory/syncStatus |

## 三、不纳入本计划的事项

- **业务功能扩展**（多任务类型、OpenAPI/SDK）：见 [产品拓展升级规划](./产品拓展升级规划.md)
- **算法模型加载治理**：见 [近期改造计划总览 §5](./近期改造计划总览.md)
- **Go/Python 端代码架构问题**：Go 端见 [Go 改造计划](./Go后端架构改造计划.md)；Python 端治理已完成并沉淀至 [Python算法服务架构文档](../04-项目实现/后端/03-Python算法服务架构文档.md)，本计划聚焦 Java，其他端问题仅在影响三端一致性时提及

## 四、经评审剔除的伪债务/低价值项

以下项经代码核实后确认不构成债务或已实现，记录于此供参考，不纳入改造：

| 原项 | 剔除原因 |
|------|---------|
| ~~评估结果永久缓存无淘汰~~ | **无法证实**：Python 应用代码 grep 不到 `{algorithmId}:{predMd5}:{refMd5}` 缓存实现，疑似基于文档臆测；且改造主体本就在 Python，不应列入 Java 计划 |
| ~~XxlJobConfig 半集成~~（原 §1.5） | **低价值**：属环境配置问题非代码架构债务，`@ConditionalOnProperty` 扫描开销可忽略 |

