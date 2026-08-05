# Flutter 端架构与 API 调用改造计划

> 本文记录 dehaze_flutter 端 API 调用层（services/models/network/providers）的深度简化改造方案。
> 目标：以真实后端接口契约为唯一依据，消除过时调用、死代码、重复逻辑与过度拆分，对齐 js SDK / Android SDK 已收敛的"薄封装 + 单一来源"架构。
> 状态：已完成（2026-08-05，`flutter analyze` 无 error）

## 一、背景与目标

dehaze_flutter 目前手写维护了一套独立的 API 调用层（7 个 Service + 13 个 Model + 网络层），但存在以下系统性问题：

- **过时调用**：部分接口在后端已不存在（如 `/algorithm-select/recommend`），部分字段/枚举与后端序列化格式不一致（状态枚举映射错位、状态值解析类型错误），导致**无法成功调用**。
- **死代码膨胀**：`ApiResult<T>` 类型从未被使用（拦截器已手工解包），大量 `ApiConstants` 定义了却无引用，多个 Service 方法（`getAlgorithmOptions`/`getAlgorithmDetail`）无调用方。
- **重复逻辑**：预测/评估两份几乎相同的轮询实现，轮询配置 `PollOptions` 定义位置随意并被跨文件 `export` 重新导出。
- **过度拆分/散落**：每个 Service 一层 `const XxxService(this._dio)` 构造 + Provider 分散在 3 个文件中，页面各写样板。
- **无用类型定义**：`ApiResult`/`PageResult` 的 json_serializable 生成物（`.g.dart`）基本未被使用；`AlgorithmRecommend` 模型字段与后端不符。

**改造原则**（与仓库已确立的规范一致，见 [文档治理规则](../00-文档治理/01-文档治理规则.md)）：

1. **接口以真实后端为准**，过时代码直接修改/删除，不做兼容兜底。
2. **单一来源**：Service 是 Flutter 侧唯一的 API 层（对应 js SDK / Android SDK 的角色），不做二次转发封装。
3. **删无用、去重复、不拆细**：能内联就内联，能一个函数解决不拆多个。

## 二、现状问题清单

### 2.1 过时 / 错误的后端调用（无法成功调用）

| # | 位置 | 问题 | 处理结果 |
|---|------|------|---------|
| 1 | `services/algorithm_service.dart` `getAlgorithmList()` | 保留 `GET /algorithms`（与 js SDK `getList()` / RN 端一致），但本地 `flatEnabledLeaves` 按 `status==1`(测试中) 过滤，**语义错误** | ✅ 保留 `GET /algorithms`（返回完整 `AlgorithmVO`，含页面所需的 `description`/`importPath`/`status`）；`flatPublishedLeaves` 改为递归收集 `status==published` 的叶子。`select/tree` 接口虽存在，但节点仅含 `id/name/type/leaf`，缺页面展示字段，切换反而需再调详情接口补字段，故不切换 |
| 2 | `services/algorithm_service.dart` `recommendAlgorithms()` | 调用 `POST /algorithm-select/recommend`，**后端不存在** | ✅ 删除；新增 `RecommendationService`：`analyze()` → `POST /recommendations/analyze`，`getRecommendations(imageMd5)` → `GET /recommendations/algorithms`（两步流程，与 RN 端一致） |
| 3 | `services/auth_service.dart` `register()` | 用 `authLogin.replaceAll('login', 'register')` 字符串 hack 拼接注册路径 | ✅ 定义独立常量 `authRegister`（`/auth/register`） |
| 4 | `models/prediction_model.dart` / `evaluation_model.dart` `TaskStatus` | `fromString` 期望 `'completed'/'failed'` 字符串，但后端 `LogStatusEnum` 以**整数**序列化（1=处理中/2=已完成/3=失败），运行时解析失败或恒为 processing | ✅ 改为整数解析（1/2/3） |
| 5 | `models/algorithm_model.dart` `AlgorithmStatus` | 枚举 `enabled=1/disabled=0/auditing=2` 与后端 6 状态（0-5）映射错位 | ✅ 对齐后端 `AlgorithmStatusEnum` 六状态（draft/testing/pendingAudit/published/disabled/archived，`@JsonValue` 0-5），`displayName` 中文映射 |
| 6 | `models/algorithm_model.dart` `importPath` | `@JsonKey(name: 'import_path')` 与后端驼峰 `importPath` 不符（build.yaml 全局 `field_rename: snake`，需显式驼峰 JsonKey 覆盖），导致 `isDeepLearning` 恒为 false | ✅ 改为 `@JsonKey(name: 'importPath')` |
| 7 | `models/recommendation_model.dart` | 新建模型未显式标注 JsonKey，被 build.yaml 全局 `field_rename: snake` 误转成 `image_md5`/`algorithm_id`，与后端驼峰不符 | ✅ 全部补显式驼峰 JsonKey（`imageUrl`/`imageMd5`/`algorithmId`/`algorithmName`/`matchScore`） |

### 2.2 无用类型定义 / 死代码

| # | 位置 | 问题 | 处理结果 |
|---|------|------|---------|
| 8 | `core/network/api_result.dart` `ApiResult<T>` | 类完全未被使用（ResponseInterceptor 与各 Service 均手工读原始 Map），`isSuccess`/`isAuthError` 等判断与拦截器重复 | ✅ 删除 `ApiResult` 类，保留 `ApiFieldError`/`ApiException`/`extractErrorMessage` |
| 9 | `core/network/page_result.dart` | 仅作 `list/total` 纯容器，`fromJson/toJson` + `.g.dart` 无调用方 | ✅ 简化为普通类，移除 json_serializable |
| 10 | `core/constants/api_constants.dart` | 大量常量无任何引用 | ✅ 仅保留实际使用的路径 |
| 11 | `models/algorithm_model.dart` `AlgorithmModel` | `config`/`remark`/`createTime`/`updateTime` 字段后端 `AlgorithmVO` 根本不存在 | ✅ 删除字段及 `algorithm_info.dart` 中对应展示（配置参数/备注/创建时间/更新时间） |
| 12 | `models/algorithm_model.dart` `AlgorithmListExtension.flatEnabledLeaves` | 仅支持两级、依赖错误的 `isEnabled` | ✅ 删除，替换为递归的 `flatPublishedLeaves` |
| 13 | `providers/processing_provider.dart` `export ... show PollOptions` | 无用的重新导出 | ✅ 删除，`PollOptions` 内聚到共享轮询工具 |

### 2.3 重复逻辑与过度拆分

| # | 位置 | 问题 | 处理结果 |
|---|------|------|---------|
| 14 | `services/prediction_service.dart` `_pollPredTask` 与 `services/evaluation_service.dart` `_pollEvalTask` | 两份几乎相同的"提交+轮询至终态"逻辑，`PollOptions` 定义在 prediction 中却被 evaluation 复用 | ✅ 提取共享轮询 helper `core/network/task_poller.dart`（`pollTask<T>` + `PollOptions`），两个 Service 复用 |
| 15 | 7 个 Service Provider 分散在 3 个文件 | `providers.dart`/`auth_provider.dart`/`processing_provider.dart` 各自定义 | ✅ Provider 统一收敛到 `providers/providers.dart`（基础设施 + 6 个服务 provider） |

### 2.4 其他

| # | 位置 | 问题 | 处理结果 |
|---|------|------|---------|
| 16 | `services/algorithm_service.dart` `getAlgorithmOptions`/`getAlgorithmDetail` | 无任何页面调用 | ✅ 删除 |

## 三、改造方案（分阶段）

### 阶段 1：修复 API 调用（对真实后端）✅

1. **算法选择**：保留 `GET /algorithms`（与 js SDK `getList()`/RN 端一致），`flatPublishedLeaves` 递归收集已发布叶子；`AlgorithmStatus` 对齐后端六状态枚举。
2. **推荐改两步流程**：`RecommendationService.analyze()` → `POST /recommendations/analyze` 获取 `imageMd5`；`getRecommendations(imageMd5)` → `GET /recommendations/algorithms`，客户端截取 TopN。删除旧 `recommendAlgorithms()` 与 `AlgorithmRecommend`。
3. **注册路径**：`api_constants.dart` 增加 `authRegister`，`AuthService.register()` 改用常量。
4. **状态解析修正**：`TaskStatus` 按整数解析（1/2/3），删除字符串 `fromString` 逻辑。
5. **字段映射修正**：`importPath`/`parentId` 及推荐模型全部显式驼峰 JsonKey，覆盖 build.yaml 的 `field_rename: snake`。

### 阶段 2：删除死代码 ✅

6. 删除 `ApiResult<T>` 类，精简 `api_result.dart`。
7. `PageResult` 简化为纯容器类，移除 `.g.dart`。
8. 清理 `ApiConstants` 未使用项。
9. 删除无调用方的 `getAlgorithmOptions()`/`getAlgorithmDetail()`、`flatEnabledLeaves`、`export ... show PollOptions`，删除后端不存在的 `AlgorithmModel` 字段（`config`/`remark`/`createTime`/`updateTime`）。

### 阶段 3：去重与收敛 ✅

10. 提取共享轮询 helper `core/network/task_poller.dart`，`PredictionService`/`EvaluationService` 复用。
11. Service Provider 统一收敛到 `providers/providers.dart`。

## 四、影响范围（调用方清单）

| 变更 | 受影响调用方 |
|------|-------------|
| `AlgorithmStatus` 六状态 + `flatPublishedLeaves`（保留 `getAlgorithmList()`） | `pages/algorithm_select/index.dart`、`pages/comparison/algorithm_info.dart` |
| 推荐两步流程 | `pages/algorithm_select/index.dart`（推荐 Tab） |
| `TaskStatus` 整数解析 | `services/prediction_service.dart`、`services/evaluation_service.dart`、`pages/processing`、`pages/comparison/metrics.dart` |
| `authRegister` 常量 | `services/auth_service.dart` |
| Provider 收敛 | `providers/providers.dart`、`providers/auth_provider.dart`、`providers/processing_provider.dart`、各页面 |
| `AlgorithmModel` 删字段（config/remark/createTime/updateTime）+ 驼峰 JsonKey | `models/algorithm_model.dart`、`models/algorithm_model.g.dart`、`pages/comparison/algorithm_info.dart`、`models/recommendation_model.dart`、`models/recommendation_model.g.dart` |

## 五、验收标准

1. ✅ 对照后端 Java 接口清单（`dehaze-java` 全部 Controller）逐一核对，Flutter 侧不再存在"后端不存在的方法/字段"。
2. ✅ `flutter analyze` 无 error。
3. ✅ 推荐流程已对齐两步（analyze → recommendations），算法选择页仅展示已发布叶子算法。
4. ✅ 预测/评估轮询状态正确（整数 1/2/3 正常流转）。
5. ✅ `ApiConstants` 无未使用常量、`models` 无未使用类型。
