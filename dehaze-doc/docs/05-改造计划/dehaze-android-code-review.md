# dehaze-android 代码审查报告

审查范围：`E:\DehazeSystem\dehaze-android`（app + sdk 模块，约 170 个 Java 源文件）
对照文档：`E:\DehazeSystem\dehaze-doc\docs`（产品概述、API 规范、模块设计）

---

## 一、冗余与重复代码

### 1.1 `UserAPI.saveToFile()` 与 `FileAPI.saveToFile()` 完全重复

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/UserAPI.java` L212-232 vs `sdk/.../api/FileAPI.java` L25-35 |
| 违反原则 | DRY（Don't Repeat Yourself） |
| 问题 | 两处实现相同的"ResponseBody → 文件"流拷贝逻辑。`TaskAPI.downloadTaskFile()` 已正确调用 `FileAPI.saveToFile()`，唯独 UserAPI 自行维护了一份私有副本，且风格不一致（手动 finally vs try-with-resources）。 |

**重构：** 删除 `UserAPI.saveToFile()`，`downloadTemplate()` 和 `export()` 内部直接调用 `FileAPI.saveToFile(response.body(), filePath)`。

---

### 1.2 Repository 层大量重复的"上传/选项/预测"包装

| 项目 | 内容 |
|------|------|
| 位置 | `CompareRepository`、`EvaluationRepository`、`PresentationRepository`、`InputHistoryRepository`、`FileRepository` 各自包含 `uploadImage()`；`CompareRepository`、`EvaluationRepository`、`PresentationRepository`、`AlgorithmRepository` 各自包含 `getAlgorithmOptions()`；`CompareRepository`、`EvaluationRepository`、`PresentationRepository` 各自包含 `getPrediction()` |
| 违反原则 | DRY / 单一职责 |
| 问题 | 同一个 SDK 调用被 3-5 个 Repository 以完全相同的方式包装。`PresentationRepository` 是 `CompareRepository` + `AlgorithmRepository` 的子集，无独立存在价值。 |

**重构：** 提取共享 Repository 或让 ViewModel 直接组合已有 Repository：

```java
// 删除 PresentationRepository，PresentationViewModel 改为：
private final AlgorithmRepository algorithmRepo = new AlgorithmRepository();
private final CompareRepository compareRepo = new CompareRepository();
```

对于 `uploadImage` / `getAlgorithmOptions` / `getPrediction` 等跨模块共用操作，归入一个 `SharedRepository`（或直接让 ViewModel 持有 `FileRepository` + `AlgorithmRepository` 引用）。

---

### 1.3 `listPredictionLogs` PageResult 拆包逻辑重复 3 次

| 项目 | 内容 |
|------|------|
| 位置 | `CompareRepository.listPredictionLogs()`、`PresentationRepository.listPredictionLogs()`、`DashboardRepository.getRecentActivities()` |
| 违反原则 | DRY |

**重构：** 在 SDK 层 `ModelAPI` 增加一个便捷方法直接返回 `List<PredictionLogVO>`（内部拆包），或在 Repository 层提取为静态工具方法。

---

### 1.4 ViewModel 回调样板代码重复 40+ 次

| 项目 | 内容 |
|------|------|
| 位置 | 所有 9 个 ViewModel 中的每个异步方法 |
| 违反原则 | DRY |
| 问题 | 每个方法都手写 `loading.setValue(true)` → `onSuccess { postValue; loading.postValue(false) }` → `onError { error.postValue; loading.postValue(false) }` 的固定模式。 |

**重构：** 在 `BaseViewModel` 中提供通用执行器：

```java
protected <T> RepositoryCallback<T> withLoading(Consumer<T> onSuccess) {
    loading.setValue(true);
    return new RepositoryCallback<T>() {
        @Override public void onSuccess(T data) {
            onSuccess.accept(data);
            loading.postValue(false);
        }
        @Override public void onError(String msg) {
            error.postValue(msg);
            loading.postValue(false);
        }
    };
}
```

调用侧简化为：`repo.getPrediction(param, withLoading(result -> predictionResult.postValue(result)));`

---

## 二、未清理的废弃逻辑与死代码

### 2.1 `AlgorithmAPI` 与 `AlgorithmSelectAPI` 功能重叠

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/AlgorithmAPI.java` L38 `compare(String ids)` / L58 `listFavorites()` / L80 `toggleFavorite(long)` vs `sdk/.../api/AlgorithmSelectAPI.java` 全部 4 个方法 |
| 违反原则 | 单一数据源 / 无死代码 |
| 问题 | 收藏和对比存在两套 API（`/api/v1/algorithms` 旧路由 vs `/api/v1/algorithm-select` 新路由），返回类型不同（`AlgorithmFavorite` vs `FavoriteVO`，`List<Algorithm>` vs `List<AlgorithmCompareVO>`）。产品文档中算法选择模块仅定义了 `/algorithm-select` 路径。 |

**重构：** 确认后端已迁移后，删除 `AlgorithmAPI` 中的 `compare()`、`listFavorites()`、`toggleFavorite()` 三个方法，统一走 `AlgorithmSelectAPI`。同步清理 `AlgorithmRepository` 中对应的 3 个方法和 `AlgorithmViewModel` 中的调用。

---

### 2.2 `AuthApiService` 缺少 `getAuthInfo()` 和 `refreshToken()` 的 Retrofit 定义

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../service/AuthApiService.java`（仅 3 个端点）vs `sdk/.../api/AuthAPI.java` L81-94 调用了 `getAuthInfo()` 和 `refreshToken()` |
| 违反原则 | 编译完整性 |
| 问题 | `AuthAPI` 调用了 `getAuthApiService().getAuthInfo()` 和 `.refreshToken()`，但 `AuthApiService` 接口中未声明这两个方法。这意味着要么存在编译错误，要么有另一个版本的文件。产品文档明确定义了 `GET /auth/me` 和 `POST /auth/refresh`。 |

**重构：** 补全 `AuthApiService`：

```java
@GET("/api/v1/auth/me")
Call<Result<AuthInfo>> getAuthInfo();

@POST("/api/v1/auth/refresh")
Call<Result<LoginResponse>> refreshToken();
```

---

### 2.3 `DehazeSDK.Builder.build()` 方法无调用者

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../DehazeSDK.java` L238-240 |
| 违反原则 | 无死代码 |
| 问题 | 实际初始化走 `DehazeSDK.initialize(builder)`（L218），`build()` 方法创建实例但不赋值给 `instance`，调用后 `getInstance()` 仍抛异常。属于误导性死代码。 |

**重构：** 删除 `build()` 方法，或将 `initialize()` 改为 `builder.build()` 内部赋值 `instance` 并返回。

---

### 2.4 `LoginViewModel` 构造函数中的冗余初始化

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/login/LoginViewModel.java` L30-39 |
| 违反原则 | 无冗余代码 |
| 问题 | `MutableLiveData` 默认值即为 null，对 `username`/`password`/`captchaCode`/`captchaKey`/`captchaImage` 设置空字符串、对 `loading`/`loginSuccess` 设置 false 是多余的——UI 层通过 `observe` 已能处理 null 初始态。 |

**重构：** 删除构造函数体，仅在需要非 null 默认值时保留（如 `loading.setValue(false)` 若 UI 依赖初始非 null）。

---

## 三、无实际意义的不必要单行函数包装

### 3.1 SDK `api/` 层约 60% 方法为零逻辑透传

| 项目 | 内容 |
|------|------|
| 位置 | `ModelAPI`（全部 6 个方法）、`DeptAPI`（5/6）、`MenuAPI`（7/8）、`RoleAPI`（7/9）、`DictAPI`（9/11）、`InputHistoryAPI`（6/9）、`AlgorithmAPI`（7/10） |
| 违反原则 | 避免无意义间接层（Law of Demeter 的反面） |
| 问题 | 典型模式：`public static void getFormData(int id, Callback cb) { service.getFormData(id).enqueue(cb); }`——无参数转换、无校验、无副作用，纯粹增加一层调用栈和维护成本。 |

**重构方案（二选一）：**

- **方案 A（推荐）：** 删除纯透传方法，Repository 层直接调用 `DehazeSDK.getInstance().getXxxApiService().method().enqueue(RepositoryAdapters.wrap(callback))`。仅保留有实际逻辑的 API 方法（DTO 构造、multipart、token 清理、文件流）。
- **方案 B：** 保留 API 层作为 SDK 对外门面，但将 Repository 层删除（因为 Repository 本身也是纯透传）。ViewModel 直接调用 API 层。

当前架构 `ViewModel → Repository → API → Service` 四层中有两层（Repository 和 API 的透传方法）完全冗余。

---

### 3.2 `LoginViewModel` 的 setter 方法

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/login/LoginViewModel.java` L43-55（`setUsername`、`setPassword`、`setCaptchaCode`） |
| 违反原则 | 避免无意义包装 |
| 问题 | 这三个方法仅调用 `liveData.setValue(value)`，而 `@Getter` 已暴露了 `MutableLiveData` 字段，UI 可直接 `viewModel.getUsername().setValue(text)`。 |

**重构：** 删除三个 setter，UI 层通过 getter 获取 LiveData 后直接设值；或改用 DataBinding 双向绑定（`android:text="@={viewModel.username}"`）彻底消除手动监听。

---

## 四、缺乏合理性的过度变量抽取

### 4.1 `DashboardRepository.getStats()` 使用数组模拟可变引用

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../repository/DashboardRepository.java` L58-60：`StatsData[] stats = new StatsData[1]; final long[] pending = {4}; final String[] firstError = {null};` |
| 违反原则 | 代码可读性 / 合理抽象 |
| 问题 | 用单元素数组绕过 lambda 的 effectively-final 限制是 Java 8 的常见 hack，但此处 4 个并发回调 + 3 个数组 + 1 个 Runnable 交织，可读性极差且存在竞态（见 5.1）。 |

**重构：** 使用 `AtomicInteger` + `AtomicReference<StatsData>` + `CountDownLatch` 或 `CompletableFuture`：

```java
public void getStats(RepositoryCallback<StatsData> callback) {
    AtomicInteger pending = new AtomicInteger(4);
    AtomicReference<StatsData> statsRef = new AtomicReference<>(new StatsData(0,0,0,0));
    AtomicReference<String> errorRef = new AtomicReference<>(null);

    BiConsumer<Integer, Long> merge = (index, value) -> {
        statsRef.updateAndGet(old -> old.withValue(index, value));
        if (pending.decrementAndGet() == 0) {
            String err = errorRef.get();
            if (err != null) callback.onError(err);
            else callback.onSuccess(statsRef.get());
        }
    };
    // ... 4 个 API 调用各自调用 merge.accept(index, count)
}
```

---

### 4.2 `CompareViewModel.predictMultiple()` 的 `int[] pending`

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/compare/viewmodel/CompareViewModel.java` L113：`final int[] pending = {algorithmIds.size()};` |
| 违反原则 | 同上 + 线程安全 |

**重构：** 改为 `AtomicInteger pending = new AtomicInteger(algorithmIds.size())`，`pending.decrementAndGet() == 0` 判断完成。

---

## 五、过度嵌套与复杂条件逻辑 / 超长函数

### 5.1 `DashboardRepository.getStats()` — 竞态条件 + 96 行

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../repository/DashboardRepository.java` L57-153 |
| 违反原则 | 单一职责 / 线程安全 |
| 问题 | (1) `checkComplete` Runnable 读取 `pending[0]` 时未加锁，但写入在 `synchronized(pending)` 内——存在可见性问题；(2) 4 个匿名内部类结构完全相同（仅 index 和 API 调用不同），占 96 行；(3) `mergeStats` 方法需要 6 个参数，其中 4 个是可变容器。 |

**重构：** 见 4.1 的 `AtomicInteger` 方案，可将 96 行压缩至约 30 行，同时消除竞态。

---

### 5.2 `CompareViewModel.predictMultiple()` — 非原子递减

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/compare/viewmodel/CompareViewModel.java` L125：`if (--pending[0] == 0)` |
| 违反原则 | 线程安全 |
| 问题 | 多个 Retrofit 回调可能在不同线程并发到达，`--pending[0]` 非原子操作，可能导致计数错乱（永远不为 0 或提前触发）。 |

**重构：** `AtomicInteger` + `decrementAndGet()`。

---

### 5.3 `UserAPI.downloadTemplate()` / `export()` — 重复的匿名 Callback 结构

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/UserAPI.java` L128-187（两个方法共 60 行，结构完全相同） |
| 违反原则 | DRY / 函数长度 |

**重构：** 提取私有方法：

```java
private static void enqueueFileDownload(Call<ResponseBody> call, String filePath, ApiCallback<Void> callback) {
    call.enqueue(new retrofit2.Callback<ResponseBody>() {
        @Override public void onResponse(...) {
            if (response.isSuccessful() && response.body() != null) {
                try { FileAPI.saveToFile(response.body(), filePath); callback.onSuccess(null); }
                catch (IOException e) { callback.onFailure(new ApiException(-1, "文件保存失败")); }
            } else {
                callback.onFailure(new ApiException(response.code(), response.message()));
            }
        }
        @Override public void onFailure(...) { callback.onFailure(new ApiException(-1, t.getMessage())); }
    });
}
```

---

## 六、违反单一职责的超长函数

### 6.1 `DashboardRepository.getStats()` — 96 行，承担 4 个 API 调用 + 并发协调 + 错误聚合

已在 5.1 给出重构方案。

### 6.2 `DatasetDetailViewModel` — 383 行，12 个结构相同的 CRUD 方法

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/dataset/DatasetDetailViewModel.java` |
| 违反原则 | 单一职责 / DRY |
| 问题 | `createItem`、`updateItem`、`deleteItem`、`batchDeleteItems`、`uploadItemFile`、`updateItemFile`、`deleteItemFile`、`batchDeleteItemFiles` 等方法结构完全一致：设 loading → 调 repository → 成功 postValue + toast + reload → 失败 postError。 |

**重构：** 使用 4.1 中 `BaseViewModel.withLoading()` + 操作后统一调用 `reloadItems()`：

```java
public void deleteItem(long itemId) {
    datasetRepo.deleteItem(itemId, withLoadingAndReload(v -> operationResult.postValue("删除成功")));
}
```

---

## 七、未对齐产品文档的冗余或缺失业务逻辑

### 7.1 登录接口 Content-Type 不符

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../service/AuthApiService.java` L22-23：`@POST` + `@Body LoginRequest`（JSON） |
| 文档要求 | API 规范 §6.1 明确：`Content-Type: application/x-www-form-urlencoded`，body 为 `username=...&password=...` |
| 影响 | 若后端严格按文档实现 form-urlencoded 解析，当前 JSON 请求将被拒绝。 |

**重构：** 改为 `@FormUrlEncoded` + `@Field` 参数，或确认后端已兼容 JSON（从 memory 中 Java 后端实际接受 JSON `LoginRequest`，此处文档与实现不一致，建议更新文档）。

---

### 7.2 Logout HTTP 方法不一致

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../service/AuthApiService.java` L36：`@POST("/api/v1/auth/logout")` |
| 文档要求 | 全局 API 规范 §6.4 定义为 `DELETE /api/v1/auth/logout`；认证模块文档定义为 `POST` |
| 影响 | 文档内部矛盾。当前实现用 POST，与 Java 后端实际路由一致（memory 确认后端为 POST）。 |

**重构：** 代码无需改动，但应更新全局 API 规范文档消除矛盾。

---

### 7.3 Dashboard 算法总数统计不准确

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../repository/DashboardRepository.java` L95-98：`AlgorithmAPI.getList(new AlgorithmQuery())` 后取 `data.size()` |
| 文档要求 | 算法列表接口 `GET /algorithms` 返回树形结构（非分页），`size()` 仅为顶层节点数，非算法总数 |
| 影响 | Dashboard 显示的"算法数量"远小于实际值。 |

**重构：** 递归统计树中所有叶子节点数量，或后端增加 `GET /algorithms/count` 端点。临时方案：

```java
private long countAlgorithms(List<Algorithm> tree) {
    if (tree == null) return 0;
    return tree.stream().mapToLong(a ->
        (a.getChildren() != null ? countAlgorithms(a.getChildren()) : 0) + 1
    ).sum();
}
```

---

### 7.4 缺失：Token 自动刷新机制

| 项目 | 内容 |
|------|------|
| 文档要求 | 认证模块 §安全要求：前端在 token 过期前 5 分钟自动刷新 |
| 当前实现 | `AuthAPI.refreshToken()` 方法存在但无任何调用者；`ApiCallback` 在收到 `A0230/A0231` 时仅清除 token，未尝试刷新 |
| 影响 | 用户每 2 小时（token 有效期）必须重新登录。 |

**重构：** 在 `ApiCallback.onResponse()` 中检测到 token 过期时，先尝试 `refreshToken()`，成功后重放原始请求；失败则清除 token 跳转登录页。或在 OkHttp Authenticator 中实现。

---

### 7.5 缺失：预测参数校验

| 项目 | 内容 |
|------|------|
| 文档要求 | 去雾处理模块：去雾强度 0-100、饱和度 0-200、对比度 0-200、锐化 0-100 |
| 当前实现 | `CompareViewModel.predict()` 和 `EvaluationViewModel` 直接将 `params` 字符串透传，无任何范围校验 |
| 影响 | 非法参数到达后端才报错，用户体验差。 |

**重构：** 在 ViewModel 或 `PredParam` 构造时校验参数范围，不合法时 `error.setValue("去雾强度须在 0-100 之间")`。

---

### 7.6 缺失：评估接口需参考图（gtUrl）

| 项目 | 内容 |
|------|------|
| 文档要求 | 评估模块：必须提供预测图 + 参考图（GT），缺少参考图返回 `B0221` |
| 当前实现 | `CompareViewModel.evaluate()` 接受 `gtUrl` 参数但未校验是否为 null；UI 层未强制要求用户上传清晰图 |

**重构：** 在 `evaluate()` 入口增加：`if (gtUrl == null || gtUrl.isEmpty()) { error.setValue("评估需要提供参考图片"); return; }`

---

## 八、缺乏类型约束的弱类型滥用

### 8.1 批量删除 ID 使用逗号拼接字符串

| 项目 | 内容 |
|------|------|
| 位置 | `AlgorithmAPI.deleteByIds(String)`、`DeptAPI.deleteByIds(String)`、`DictAPI.deleteDictTypes(String)`、`DictAPI.deleteDictByIds(String)`、`RoleAPI.deleteByIds(String)`、`UserAPI.deleteByIds(String)` |
| 违反原则 | 类型安全 |
| 问题 | 调用方需手动 `"1,2,3"` 拼接，无编译期校验，易产生 `"1,,2"` 或空串等运行时错误。 |

**重构：** API 签名改为 `List<Long>` / `List<Integer>`，内部拼接：

```java
public static void deleteByIds(List<Long> ids, ApiCallback<Void> callback) {
    String joined = ids.stream().map(String::valueOf).collect(Collectors.joining(","));
    service.deleteAlgorithms(joined).enqueue(callback);
}
```

---

### 8.2 状态字段使用魔法数字

| 项目 | 内容 |
|------|------|
| 位置 | `RoleAPI.updateStatus(long id, int status)`、`UserAPI.updateStatus(long id, int status)`、`MenuAPI.updateVisible(long id, int visible)` |
| 违反原则 | 类型安全 / 可读性 |
| 问题 | `0`/`1` 含义不明确，调用侧 `updateStatus(id, 1)` 无法表达意图。 |

**重构：** 定义枚举或至少使用 `@IntDef` 注解：

```java
@IntDef({Status.ENABLED, Status.DISABLED})
@Retention(RetentionPolicy.SOURCE)
public @interface Status { int ENABLED = 1; int DISABLED = 0; }
```

---

### 8.3 数据集图片类型使用裸字符串

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../api/DatasetAPI.java` `uploadItemFile(..., String type, ...)` |
| 文档定义 | 类型为 `clear`/`hazy`/`trans`（固定枚举） |

**重构：** 定义 `ImageType` 枚举（`CLEAR("clear"), HAZY("hazy"), TRANS("trans")`），API 签名改为 `ImageType type`。

---

### 8.4 `multiPredictionResults` 使用 `Map<String, PredResult>` 而非 `Map<Long, PredResult>`

| 项目 | 内容 |
|------|------|
| 位置 | `app/.../ui/compare/viewmodel/CompareViewModel.java` L30、L123 |
| 违反原则 | 类型安全 |
| 问题 | key 是 algorithmId（long），却转为 String 存储，读取时需反向解析，无编译期保护。 |

**重构：** 改为 `MutableLiveData<Map<Long, PredResult>>`，`results.put(algorithmId, result)`。

---

### 8.5 `TokenManager` 同时使用 `volatile` 和 `synchronized`

| 项目 | 内容 |
|------|------|
| 位置 | `sdk/.../utils/TokenManager.java` L5：`private static volatile String token` + 所有方法 `synchronized` |
| 违反原则 | 冗余语义 |
| 问题 | 所有读写均在 `synchronized(TokenManager.class)` 内，`volatile` 完全多余，误导读者以为存在非同步访问路径。 |

**重构：** 移除 `volatile` 关键字。

---

## 九、架构级问题汇总

### 9.1 四层透传架构

当前调用链：`ViewModel → Repository → API(static) → Service(Retrofit)`

其中 Repository 层 18 个类中，15 个是纯一行透传（仅 `DashboardRepository`、`FileRepository`、`TaskRepository` 有少量逻辑）。API 层 13 个类中约 60% 方法也是纯透传。

**建议：** 合并为两层 `ViewModel → API → Service`，将 Repository 中仅有的逻辑（PageResult 拆包、文件保存路径）上移到 ViewModel 或下沉到 API 层。若坚持 Repository 层，则删除 API 层的透传方法，Repository 直接持有 Service 引用。

### 9.2 `LoginViewModel` 绕过 Repository 层

`LoginViewModel` 直接调用 `AuthAPI`（SDK 层），是唯一不走 Repository 的 ViewModel。应补充 `AuthRepository` 或统一改为直接调用 API 层（若按 9.1 删除 Repository）。

### 9.3 私有构造函数不一致

`AlgorithmSelectAPI`、`AuthAPI`、`FileAPI`、`InputHistoryAPI`、`TaskAPI` 有私有构造函数；`AlgorithmAPI`、`DatasetAPI`、`DeptAPI`、`DictAPI`、`MenuAPI`、`ModelAPI`、`RoleAPI`、`UserAPI` 没有。所有都是纯静态工具类，应统一添加。

---

## 十、优先级排序

| 优先级 | 问题编号 | 影响 |
|--------|----------|------|
| P0（功能缺陷） | 7.4 Token 无刷新 | 用户每 2h 强制登出 |
| P0（功能缺陷） | 5.1/5.2 竞态条件 | 并发回调时 UI 状态错乱 |
| P0（功能缺陷） | 7.3 算法计数错误 | Dashboard 数据失真 |
| P1（文档对齐） | 7.1 登录 Content-Type | 潜在兼容风险 |
| P1（文档对齐） | 7.5/7.6 参数校验缺失 | 用户体验差 |
| P2（代码质量） | 1.1-1.4 重复代码 | 维护成本高 |
| P2（代码质量） | 2.1 废弃 API 未清理 | 混淆调用方 |
| P2（代码质量） | 3.1 无意义间接层 | 架构臃肿 |
| P3（类型安全） | 8.1-8.5 弱类型 | 运行时错误风险 |
| P3（一致性） | 9.2/9.3 风格不统一 | 可读性 |
