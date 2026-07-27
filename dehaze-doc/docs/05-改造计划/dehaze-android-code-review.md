# dehaze-android 待修复问题清单

> 本文档为代码审查报告中**未修复问题**的追踪清单，已修复问题已删除。审查范围：`E:\DehazeSystem\dehaze-android`（app + sdk 模块）。


## 二、文档对齐（P1）

### 2.1 登录接口 Content-Type 不符

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../service/AuthApiService.java` L15-16：`@POST` + `@Body LoginRequest`（JSON） |
| 文档要求 | API 规范 §6.1 明确：`Content-Type: application/x-www-form-urlencoded`，body 为 `username=...&password=...` |
| 影响 | 若后端严格按文档实现 form-urlencoded 解析，当前 JSON 请求将被拒绝 |

**修复方案**：
- 方案 A：代码改为 `@FormUrlEncoded` + `@Field` 参数
- 方案 B：确认后端已兼容 JSON（Java 后端实际接受 JSON `LoginRequest`），更新 API 规范文档消除矛盾

### 2.2 预测参数校验未全覆盖

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/compare/viewmodel/EvaluationViewModel.java` L69 |
| 当前实现 | `DehazeParams.validate()` 已实现（校验去雾强度 0-100、饱和度 0-200、对比度 0-200、锐化 0-100）；`CompareViewModel.predict()` 已调用 `params.validate()`；但 `EvaluationViewModel` L69 使用默认 `new DehazeParams()` 未触发用户参数校验路径 |
| 影响 | 评估场景下参数范围校验未生效 |

**修复方案**：`EvaluationViewModel` 应使用用户传入的 params 并调用 `validate()`，而非默认值。

---

## 三、废弃逻辑未清理（P2）

### 3.1 AlgorithmAPI 与 AlgorithmSelectAPI 功能重叠

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/AlgorithmAPI.java` L42 `compare(String ids)` / L62 `listFavorites()` / L84 `toggleFavorite(long)` vs `sdk/.../api/AlgorithmSelectAPI.java` 全部 4 个方法 |
| 问题 | 收藏和对比存在两套 API（`/api/v1/algorithms` 旧路由 vs `/api/v1/algorithm-select` 新路由），返回类型不同（`AlgorithmFavorite` vs `FavoriteVO`，`List<Algorithm>` vs `List<AlgorithmCompareVO>`）。产品文档中算法选择模块仅定义了 `/algorithm-select` 路径 |
| 影响 | 调用方混淆，维护两套等价逻辑 |

**修复方案**：确认后端已迁移后，删除 `AlgorithmAPI` 中的 `compare()`、`listFavorites()`、`toggleFavorite()` 三个方法，统一走 `AlgorithmSelectAPI`。同步清理 `AlgorithmRepository` 中对应的 3 个方法和 `AlgorithmViewModel` 中的调用。

### 3.2 SDK 四层透传架构冗余

| 项目 | 内容 |
|------|------|
| 位置 | `ViewModel → Repository → API(static) → Service(Retrofit)` 四层 |
| 问题 | Repository 层 18 个类中，15 个是纯一行透传（仅 `DashboardRepository`、`FileRepository`、`TaskRepository` 有少量逻辑）。API 层 13 个类中约 60% 方法也是纯透传 |
| 现状 | `ModelAPI` 已增加 `predictAndWait`/`evaluateAndWait`/轮询逻辑等非透传代码；`DeptAPI`/`RoleAPI` 等已包含 query 参数提取与 ID 列表拼接逻辑。但 Repository 仍多为薄包装 |

**修复方案（二选一）**：
- 方案 A（推荐）：删除纯透传的 Repository 层，ViewModel 直接调用 API 层
- 方案 B：保留 Repository 层，删除 API 层的透传方法，Repository 直接持有 Service 引用

---

## 四、类型安全（P3）

### 4.1 批量删除 ID 使用逗号拼接字符串

| 项目 | 内容 |
|------|------|
| 位置 | `AlgorithmAPI.deleteByIds(String)`、`DeptAPI.deleteByIds(String)`、`DictAPI.deleteDictTypes(String)`、`DictAPI.deleteDictByIds(String)`、`RoleAPI.deleteByIds(String)`、`UserAPI.deleteByIds(String)` |
| 问题 | 调用方需手动 `"1,2,3"` 拼接，无编译期校验，易产生 `"1,,2"` 或空串等运行时错误 |

**修复方案**：API 签名改为 `List<Long>` / `List<Integer>`，内部拼接：
```java
public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
    String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
    service.deleteAlgorithms(joined).enqueue(callback);
}
```

### 4.2 状态字段使用魔法数字

| 项目 | 内容 |
|------|------|
| 位置 | `RoleAPI.updateStatus(long id, int status)`、`UserAPI.updateStatus(long id, int status)`、`MenuAPI.updateVisible(long id, int visible)` |
| 问题 | `0`/`1` 含义不明确，调用侧 `updateStatus(id, 1)` 无法表达意图 |

**修复方案**：定义枚举或至少使用 `@IntDef` 注解：
```java
@IntDef({Status.ENABLED, Status.DISABLED})
@Retention(RetentionPolicy.SOURCE)
public @interface Status { int ENABLED = 1; int DISABLED = 0; }
```

### 4.3 数据集图片类型使用裸字符串

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/DatasetAPI.java` `uploadItemFile(..., String type, ...)` |
| 文档定义 | 类型为 `clear`/`hazy`/`trans`（固定枚举） |

**修复方案**：定义 `ImageType` 枚举（`CLEAR("clear"), HAZY("hazy"), TRANS("trans")`），API 签名改为 `ImageType type`。

### 4.4 `multiPredictionResults` 使用 `Map<String, PredResult>` 而非 `Map<Long, PredResult>`

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/compare/viewmodel/CompareViewModel.java` L30、L123 |
| 问题 | key 是 algorithmId（long），却转为 String 存储，读取时需反向解析，无编译期保护 |

**修复方案**：改为 `MutableLiveData<Map<Long, PredResult>>`，`results.put(algorithmId, result)`。

### 4.5 `TokenManager` 同时使用 `volatile` 和 `synchronized`

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../utils/TokenManager.java` L5：`private static volatile String token` + 所有方法 `synchronized` |
| 问题 | 所有读写均在 `synchronized(TokenManager.class)` 内，`volatile` 完全多余，误导读者以为存在非同步访问路径 |

**修复方案**：移除 `volatile` 关键字。
