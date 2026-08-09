# Uniapp 前端架构改造计划

> 适用项目：`dehaze-uniapp`
> 依据：对 `dehaze-uniapp/src` 实际代码的逐层核验（pages.json / manifest.json / package.json / store / layout / api / 14 个 system 页面 / dehaze·processing·batch 等核心页面）
> 定位：本计划仅收录**经代码核验确认存在、且改动具有明确架构价值**的问题，不收录主观风格偏好与未验证的猜测。

## 一、现状评估

整体架构骨架（PageLayout L0–L3 分层、Pinia + SDK 适配器、pages.json 单一路由源、视角拆分）方向正确且与设计文档一致。但在**实现一致性、公共抽象、依赖治理**三个维度存在若干结构性问题，部分已影响可运行性与可维护性，需分阶段治理。

## 二、改造项

### 2.1【P0·阻断性】uview-plus 组件残留与依赖事实缺失

**问题**

- 全局 48 个文件仍使用 `<u-*>` / `<up-*>` 组件（`u-form` / `u-input` / `u-button` / `u-switch` / `u-popup` / `up-loading-icon` 等），代表文件：[system/user/detail.vue](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/system/user/detail.vue)、[processing/index.vue](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/processing/index.vue)（`<up-loading-icon>`）。
- 但 `package.json` 依赖中**已无 uview-plus**，`pages.json` 的 `easycom.custom` 也**未配置** `u-*` / `up-*` 规则。
- 架构文档 §10 已声明"已移除 uview-plus，仅保留业务自建组件"。

**影响**：组件被引用却无解析来源，属"声明与实现相悖"的阻断性缺陷——要么运行期渲染失败，要么依赖未声明的隐式解析，构建/升级时极易崩溃。

**改造**

1. 全量盘点 48 个文件中 `u-*` / `up-*` 的使用清单与用途（表单/按钮/弹窗/开关/加载）。
2. 优先 uni 原生方案，避免重造 uview-plus：
   - 表单类（`u-form`/`u-form-item`/`u-input`/`u-switch`）→ uni 原生 `input`/`switch` + 原生 `form`/`cell` 布局；表单字段包装仅在确认多处复用后才抽 `FormField`。
   - `u-button` → 原生 `button` + 全局按钮样式类。
   - `u-popup` → 优先 `uni-popup`（uni-app 官方插件），不另起 `Popup` 组件。
   - `up-loading-icon` → 优先 CSS 旋转就地实现或复用已有 `SvgIcon`，不新增 `LoadingIcon` 组件。
3. 替换后移除一切 `u-*`/`up-*` 残留

### 2.2【P0·正确性】迁移残留的死路由 `/pages/user-center/index`

**问题**

- 2 处硬编码跳转到已不存在的旧路径：
  - [dehaze/index.vue:510](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/dehaze/index.vue#L510)（配额不足"去充值"）
  - [processing/index.vue:338](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/processing/index.vue#L338)（同上）
- 实际第 5 Tab 路径为 `pages/profile/index`（见 pages.json tabBar）。

**影响**：用户点击"去充值"跳转失败，核心付费链路断裂。

**改造**：两处统一改为 `/pages/personal/quota/index`（额度页）或 `/pages/personal/orders/index`（订单页），与"充值"语义对齐。仅 2 处引用，直接修改路径即可，不另抽 ROUTE 常量集合。

### 2.3【P1·重复逻辑】去雾处理主链路逻辑散落页面、跨页重复

**问题**

- 配额检查 + 确认弹窗 + `ModelAPI.predictAndWait` + 重试 + 耗时计时 的处理逻辑，在 [dehaze/index.vue](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/dehaze/index.vue#L495)（L495 起）与 [processing/index.vue](file:///E:/DehazeSystem/dehaze-uniapp/src/pages/processing/index.vue#L272)（L272 起）**各实现一套**，差异仅在 UI 呈现。
- `processing` store（[store/processing.ts](file:///E:/DehazeSystem/dehaze-uniapp/src/store/processing.ts)）只存状态，不承载"执行处理"这一核心业务动作。
- `batch/index.vue` 又有第三套 `predictAndWait` 轮询实现（L335、L397）。

**影响**：处理逻辑三处分散，重试策略（RETRY_DELAYS）、配额校验、取消语义任一变更需改三处，易漏改。

**改造**：将"配额校验 + 提交预测 + 重试 + 取消 + 耗时"收敛进 `useProcessingStore` 的一个 action（如 `runPrediction(opts)`），页面只负责 UI 绑定与回调；批量处理复用同一 action 的单次执行能力。`dehaze` 为 Tab 页有独立交互流，**保留 dehaze 页 UI**，仅将处理逻辑改为调用 store action，不改为跳转 `processing` 页，避免改变用户操作路径。

### 2.4【P1·公共抽象缺失】system 模块 CRUD 样板大面积复制

**问题**（经抽样 user/role/dict/order 核验）

- 分页样板（`list/keyword/pageNum/hasMore/loading` + `fetchList(reset)` + `loadMore`）在 user/order 间逐行重复，全目录 `fetchList/loadMore/pageNum/hasMore` 命中 12 文件 111 处。
- 删除确认流程（`showModal` → `deleteByIds` → `showToast` → `fetchList`）在 user/role 间近乎逐字复制。
- 表单弹窗实现方式不统一：dict 用内联 `u-popup`，user/role 用独立 `detail.vue` 跳转，两套模式并存。
- `.fab-btn` 样式块在 7 个文件逐字复制。
- `src/**/{composables,hooks}` 目录**不存在**，无任何 `useList`/`usePagination` 抽象。

**改造**

1. 新增 `src/composables/usePagedList.ts`：封装分页 state + `fetchList(reset)` + `loadMore` + `handleSearch`，按 `(fetcher, options)` 参数化。
2. `useCrudDelete` **先评估再定**：当前删除确认流程（`showModal` → `deleteByIds` → `showToast` → `fetchList`）仅 user/role 两处逐字复制（各约 14 行），task 的 `cancelTask` 为"取消"语义、结构相似但动作不同。按"仅 1-2 处复用偏过度设计"原则，需先确认除 user/role/task 外是否还有同模式调用点，若复用度不足则就地保留、不抽 composable。
3. 将 `.fab-btn` 提取为 `components/common/FabButton.vue`。
4. 统一表单编辑入口策略：弹窗 vs 跳页二选一（建议弹窗用于轻量字典类，跳页用于复杂表单，在文档中明确边界）。

### 2.5【P1·安全/规范】system 模块权限控制大面积缺失

**问题**

- 14 个 system 子页面中仅 `algorithm/index.vue` 使用 `hasPerm`（4 处：add/audit/edit/delete），其余 13 个页面的删除/新增/审核/退款按钮（如 order 的 approveRefund/rejectRefund）**无任何权限判断**。role/index、role/detail 此前误报使用 hasPerm，实际 grep 仅命中 `/role/permission` 路由路径字符串，并非权限判断调用。
- 架构文档 §6 声明"页面级和操作级权限判断"，但实现未落地。

**影响**：低权限用户可执行管理操作，属安全缺陷（虽有后端兜底，但前端不应放任）。

**改造**：抽取 `v-perm` 指令或 `<PermButton>` 组件（基于 auth store `hasPerm`），对 system 模块所有写操作按钮统一加权限码绑定；补全 13 个缺失页面的按钮级权限。

### 2.6【P1·设计令牌未落地】硬编码颜色泛滥

**问题**

- [styles/variables.scss](file:///E:/DehazeSystem/dehaze-uniapp/src/styles/variables.scss) 已定义完整 `$color-*` 令牌，但 pages 下 **465 处** `background:#xxx` / `color:#xxx` 硬编码，分布于 58 个文件（如 processing/index.vue 29 处、algorithm-select/index.vue 57 处、task-history 17 处）。
- 文档 §3.5 / §10 声明"复用 variables.scss 令牌"，实现严重背离。

**改造**：分批将硬编码色值映射到 `$color-*` 令牌；对令牌未覆盖的语义色（如 processing 的 `#f59e0b` 系列）补充令牌定义。优先改造高频复用页面（processing/algorithm-select/system）。可加一条 stylelint 规则禁止 `style` 块内裸 hex 色（令牌定义文件除外）。

### 2.7【P2·重复实现】状态栏适配逻辑与安全区样式重复

**问题**

- `Navbar.vue` 与 `ImmersiveLayout.vue` 各自 `onMounted` 重复 `uni.getSystemInfoSync().statusBarHeight` 取值逻辑（[Navbar.vue:72](file:///E:/DehazeSystem/dehaze-uniapp/src/layout/Navbar.vue#L72)、[ImmersiveLayout.vue:52](file:///E:/DehazeSystem/dehaze-uniapp/src/layout/ImmersiveLayout.vue#L52)）。
- 状态栏占位用 `height: statusBarHeight + 'px'`（**px**），而文档 §3.5 声明全局 rpx、仅 1px 边框/@media 保留 px。状态栏高度用 px 是合理的（系统返回值即 px），但需在文档中显式说明此例外，避免误判为违规。
- `padding-bottom: calc(Xrpx + env(safe-area-inset-bottom))` 安全区样板在 layout/index、file-manage、task-history、dataset 等多处重复，且 `variables.scss` 已定义 `$safe-area-bottom*` 令牌却未被使用。

**改造**：状态栏取值逻辑仅 3 行 + 仅 2 处复用（Navbar/ImmersiveLayout），按"仅 1-2 行且缺乏复用场景不应抽取为独立函数"原则，**不抽 `useStatusBar` composable**，维持现状即可；安全区底部 padding 抽取为 SCSS `@mixin safe-area-bottom($base)` 并复用 `$safe-area-bottom*` 令牌（13 处复用充分，值得抽取），消除各页面重复 calc。

### 2.8【P2·网络层一致性】文件上传绕过 SDK 响应/错误处理

**问题**

- [api/file.ts](file:///E:/DehazeSystem/dehaze-uniapp/src/api/file.ts) 的 `uploadImage` 自行从 storage 读 session 注入 header、自行判断 `code !== ResultEnum.SUCCESS`、自行抛错，**绕过了 SDK 的 axios 拦截器链**（会话失效重登、trace_id、统一错误格式均不生效）。
- 理由（小程序端 `uni.uploadFile` 不走 axios）成立，但当前实现未与 SDK 的会话失效事件 `SESSION_INVALID_EVENT` 对接：上传接口返回 A0230/A0231 时不会触发重登。

**改造**：在 `uploadImage` 的响应处理中，对失效错误码（A0230/A0231/A0301）复用 `sdk-setup.ts` 的 `redirectToLogin` 逻辑（抽出为公共函数），确保上传链路与普通请求链路的会话失效行为一致；session 注入改为从 auth store 读取而非直接 storage，保持单一数据源。

## 三、优先级与建议顺序

| 优先级 | 改造项 | 理由 |
|--------|--------|------|
| P0 | 2.1 uview-plus 残留 | 影响可运行性，声明与实现冲突 |
| P0 | 2.2 死路由 user-center | 付费链路断裂，改动极小 |
| P1 | 2.3 处理主链路收敛 | 核心业务逻辑去重 |
| P1 | 2.4 system CRUD 抽象 | 消除 40–50% 样板，提升可维护性 |
| P1 | 2.5 system 权限补全 | 安全缺陷 |
| P1 | 2.6 设计令牌落地 | 视觉一致性与规范遵从 |
| P2 | 2.7 安全区样式去重 | 局部整洁度 |
| P2 | 2.8 上传链路会话一致性 | 边界场景健壮性 |

建议 P0 立即处理；P1 按 2.3 → 2.4 → 2.6 → 2.5 顺序推进（2.4 抽象落地后再补 2.5 权限封装更顺）；P2 随相关模块改动顺带完成，不单独立项。

## 四、不纳入本计划的事项

- 路由守卫、SDK 适配器、Pinia 模块划分等骨架设计：经核验与文档一致，无需改造。
- `uni-adapter.ts` 的 PATCH→POST 回退：属合理的多端兼容处理，保留。
- 各业务页面的具体交互细节：属需求/前端实现文档范畴，非架构层问题。
