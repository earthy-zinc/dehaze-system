# Web 前端架构改造计划

> 本文档聚焦 `dehaze-front-vue` 与 `dehaze-front-react` 两个 Web 前端项目在**代码架构层面**的实际问题与改造方向，供后续重构参考。架构文档失真问题（WebSocket 虚假描述、favorite Store 不存在等）已在 [01-Vue架构文档.md](../04-项目实现/前端/01-Vue架构文档.md)、[02-React架构文档.md](../04-项目实现/前端/02-React架构文档.md) 修复中处理，本文不重复。
>
> 筛选原则：仅纳入有明确可维护性/可靠性/一致性收益且改造成本合理的项。风格偏好类问题（如 store 导出命名风格）、trivial 配置错误（如 `productName` 残留 vue 字样）、纯类型标注缺失等不单列改造项，在相关重构中顺带处理。
>
> 共性总纲：改造项需以真实业务需求与数据量为前提，避免为一致性而过度设计（呼应项目通用规则"禁止过度设计"）。评审中已据此降级或剔除静态路由动态化、notification slice、推荐 hook、虚拟滚动等项。

## 一、问题总览

| # | 问题 | 类别 | 优先级 | 影响范围 |
|---|------|------|:------:|----------|
| 1 | React `favoriteSlice.toggleFavorite` 不调用 API，收藏功能核心失效 | 功能缺陷 | P0 | React 收藏全流程 |
| 2 | 死代码批量清理：Vue 5 个 composable + Magnifier 双版本 + React datasetSlice + Settings 空文件 | 可维护性 | P0 | 两端多处 |
| 3 | Vue `dehazeParams` 三处定义，数据源不统一 | 数据一致性 | P1 | Vue 去雾/对比/评估 |
| 4 | Vue `imageShow` store 20+ 纯赋值 setter 过度封装 | 可维护性 | P1 | Vue store/modules/imageShow.ts |
| 5 | 两端列表页 CRUD 样板重复，已有 composable/hook 抽象却零引用 | 可维护性 | P1 | 两端 20+ 列表页 |
| 6 | React 错误处理双重提示（全局拦截器 + 组件 catch 各弹一次） | 用户体验 | P1 | React 66 个文件 |
| 7 | React 缺失 i18n，语言切换仅切 antd locale，应用文本硬编码中文 | 功能完整性 | P1 | React 全局 |
| 8 | 两端 `dataset/list/detail` 巨型组件（Vue 1612 行 / React 1700 行）职责过载 | 可维护性 | P2 | 两端数据集详情页 |
| 10 | Store slice 两端不对齐：React 缺 algorithm/tagsView（notification 未读数留组件、favorite 待修） | 跨端一致性 | P2 | 两端 store/modules |
| 13 | 核心模块零测试：两端权限守卫、动态路由、轮询状态机均无覆盖 | 质量 | P2 | 两端核心逻辑 |
| 14 | React `imageShowSlice` 持久化了 loading/mouse/mask 等临时状态 | 健壮性 | P3 | React imageShowSlice |
| 15 | React 缺 TagsView 组件与动态 Breadcrumb | 功能完整性 | P3 | React layout |
| 16 | Vue `task` store 内含 DOM 下载操作，职责越界 | 可维护性 | P3 | Vue store/modules/task.ts |

---

## 二、P0：功能缺陷与死代码

### 2.1 React 收藏功能核心失效

**现状**：`favoriteSlice.toggleFavorite` async thunk 内部不调用任何 `FavoriteAPI`，直接返回硬编码 `{ isFavorited: false }`。且返回字段名 `isFavorited` 与 `useFavorite` 读取的 `result?.favorited` 不匹配，导致条件恒为 false。

**证据**：
- [favoriteSlice.ts](file:///e:/DehazeSystem/dehaze-front-react/src/store/modules/favoriteSlice.ts) 第 45-57 行：thunk 体仅 `return { targetType, targetId, isFavorited: false }`
- [useFavorite.ts](file:///e:/DehazeSystem/dehaze-front-react/src/hooks/useFavorite.ts) 第 28 行：检查 `result?.favorited`（字段名不一致）
- `extraReducers` 中 `toggleFavorite.fulfilled` 执行 `!isFavorited`，由于恒为 false，每次 toggle 只会把状态设为 true，**无法取消收藏**

**影响**：收藏功能核心交互完全失效——用户无法取消收藏，收藏状态不持久化到后端。

**改造方向**：thunk 内部根据当前状态调用 `FavoriteAPI.add()` 或 `FavoriteAPI.deleteByIds()`，返回真实 favorited 状态，并统一字段名。同时补 `favoriteSlice` 的单元测试覆盖 toggle 逻辑。

### 2.2 死代码批量清理

**现状**：两端存在多处已废弃但未删除的代码，形成"有抽象 + 仍重复"的最差组合，误导维护者。

| 项目 | 死代码 | 证据 |
|------|--------|------|
| Vue | 5 个 composable 零引用：`useAsyncTask`、`useDebounce`、`useDebouncedRef`、`useTableSelection`、`useConfirm`/`useDeleteConfirm` | 全局搜索仅命中自身定义；项目实际用 vueuse `useDebounceFn` |
| Vue | `Magnifier/index.vue`（206 行）无生产引用，实际用 `newIndex.vue` | [AlgorithmToolBar/index.vue:2](file:///e:/DehazeSystem/dehaze-front-vue/src/components/AlgorithmToolBar/index.vue) 仅导入 newIndex |
| React | `datasetSlice.ts` 定义完整 CRUD thunk 但零调用 | [dataset/list/index.tsx:65](file:///e:/DehazeSystem/dehaze-front-react/src/pages/dataset/list/index.tsx) 直接调用 `DatasetAPI.getList` 绕过 store |
| React | `components/Settings/index.tsx` 内容仅 `export {};` | 实际 Settings 组件在 `layout/components/NavBar/Settings.tsx` |

**影响**：死代码增加心智负担，`datasetSlice` 持久化的 `datasetList` 还会占用 localStorage 却不被消费。

**改造方向**：直接删除上述死代码。Vue 端防抖统一用 vueuse `useDebounceFn`；删除 `Magnifier/index.vue` 后将 `newIndex.vue` 重命名为 `index.vue` 并更新测试；React 端从 `store/index.ts` 移除 `dataset` reducer。

---

## 三、P1：数据流与状态管理

### 3.1 Vue dehazeParams 三处定义

**现状**：去雾算法参数 `dehazeParams`（dehazeStrength/colorSaturation/contrast/sharpen）在 store 和两个组件中各定义一份，默认值相同但互不同步。

**证据**：
- [imageShow.ts:78-83](file:///e:/DehazeSystem/dehaze-front-vue/src/store/modules/imageShow.ts) store 内 reactive 定义
- [presentation/dehaze/index.vue:36-41](file:///e:/DehazeSystem/dehaze-front-vue/src/views/presentation/dehaze/index.vue) 组件本地 ref 定义
- [AlgorithmToolBar/index.vue:43](file:///e:/DehazeSystem/dehaze-front-vue/src/components/AlgorithmToolBar/index.vue) 组件本地 reactive 定义

store 中的 `dehazeParams` 实际只被 `evaluation/index.vue` 读取，对真正的去雾页面毫无作用，store 字段形同虚设。

**影响**：同一业务概念存在 3 个数据源，参数实际取值取决于页面，存在数据不一致 bug 隐患。

**改造方向**：统一为单一数据源。根据当前使用模式，应删除 store 中的 `dehazeParams`，由 `AlgorithmToolBar` 通过 emit 向父级传递参数，evaluation 页面若需共享则通过路由参数或 props 传入。

### 3.2 Vue imageShow store 过度封装

**现状**：`imageShow` store 暴露 20+ 个仅 1 行的纯赋值 setter，违反项目规则"禁止无意义的过度封装：仅 1-2 行且缺乏复用场景的代码不应抽取为独立函数"。Pinia 允许直接修改 state，这些 setter 没有任何附加逻辑。

**证据**：[imageShow.ts](file:///e:/DehazeSystem/dehaze-front-vue/src/store/modules/imageShow.ts) 第 89-186 行共 13 个纯赋值 setter（`setBrightness`/`setContrast`/`setMagnifierShow`/`setMagnifierShape` 等），return 块暴露 21 个方法。

**影响**：store 文件膨胀，调用方需记忆大量无意义方法名。

**改造方向**：删除所有纯赋值 setter，调用方直接 `store.imageInfo.brightness = v`。仅保留含业务逻辑的方法（如 `setImageUrl` 含 label 颜色逻辑）。

### 3.3 两端列表页 CRUD 样板重复

**现状**：两端 20+ 个列表页重复实现相同的 CRUD 模式（`openDialog`/`handleSubmit`/`closeDialog`/`resetForm`/`handleDelete` + `ElMessageBox.confirm`/`message.error` 删除确认），且两端都已实现 `usePagination`/`useTableSelection` 抽象但**零引用**。

**证据**：
- Vue：8+ 列表页重复，`useConfirm`/`useDeleteConfirm` 全局零调用
- React：20 个列表页重复 `useState(loading)`/`useState(pageData)`/`useState(total)`/`useState(queryParams)`/`useState(selectedRowKeys)`/`useState(refreshFlag)`/`loadData`/`refreshList`，`usePagination`/`useTableSelection` 零调用

**影响**：删除确认文案、按钮文字、行为在每个页面各写一遍，难以统一。

**改造方向**：
1. 启用已有 `usePagination`/`useTableSelection` hook 组合消除列表数据流样板，不再向上抽取 `useTablePage<T>` 超抽象（避免违反"禁止过度设计"规则）
2. Vue 端启用 `useDeleteConfirm` 替换所有 `ElMessageBox.confirm` 删除确认
3. React 端将 Dialog/Drawer 拆为独立组件（参照 Vue 端已有拆分或统一规范）

> 注意：列表页局部数据不强制进 store，组件本地状态管理是合理的架构选择。改造重点在于消除重复样板，而非改变数据流分层。

### 3.4 React 错误处理双重提示

**现状**：全局 axios 拦截器 `onResponseError` 已调用 `message.error()`，组件 catch 块又调用 `message.error()`，用户每次遇到 API 错误看到两次错误提示。

**证据**：
- [request.ts:11](file:///e:/DehazeSystem/dehaze-front-react/src/utils/request.ts) 全局拦截器 `message.error(msg || "系统出错")`
- [RefundAuditDialog.tsx:69](file:///e:/DehazeSystem/dehaze-front-react/src/pages/order/refund/components/RefundAuditDialog.tsx) `message.error(error?.message || "操作失败")`
- 全项目 66 个文件包含 `message.error` 调用

**改造方向**：组件 catch 块不再调用 `message.error`，仅处理业务逻辑（如重置 loading 状态）。全局拦截器负责统一错误提示。

### 3.5 React 缺失 i18n

**现状**：Vue 端有完整 i18n（`lang/package/en.ts`、`zh-cn.ts`、`plugins/i18n.ts`），React 端完全没有。Settings 面板有语言切换选项但仅切换 antd 的 locale，应用文本全部硬编码中文。

**证据**：React 端搜索 `i18n|useTranslation` 零匹配；[Settings.tsx:44-54](file:///e:/DehazeSystem/dehaze-front-react/src/layout/components/NavBar/Settings.tsx) 语言切换仅作用于 antd。

**影响**：英文用户看到中文界面 + 英文 antd 组件的混合状态。

**改造方向**：i18n 改造成本高、价值取决于是否存在英文用户业务需求，**非纯技术重构**。需先确认产品层面是否需要多语言支持再立项；若确认立项，引入 `react-i18next`，复用 Vue 端语言资源结构，抽取文案 key，从 layout 和核心页面开始逐步覆盖。

---

## 四、P2：组件拆分与跨端对齐

### 4.1 巨型组件拆分

**现状**：两端 `dataset/list/detail` 均超 1600 行，单组件承担 9-10 类职责。

| 项目 | 文件 | 行数 |
|------|------|------|
| Vue | [dataset/list/detail.vue](file:///e:/DehazeSystem/dehaze-front-vue/src/views/dataset/list/detail.vue) | 1612 |
| React | [dataset/list/detail/index.tsx](file:///e:/DehazeSystem/dehaze-front-react/src/pages/dataset/list/detail/index.tsx) | 1700 |

Vue 端另有 4 个超 1000 行的 view：`member/list`(1037)、`package/list`(1033)、`presentation/dehaze`(1015)、`feedback/list`(877，内联 5 个弹窗)。

**改造方向**：按职责拆分为子组件。以 dataset/detail 为例：
- 展示模式拆为 `ListView`/`WaterfallView`/`GridView`
- 图片预览弹窗独立为 `ImagePreviewDialog`
- 配对上传/批量上传独立为 `PairUploadDialog`/`BatchUploadDialog`
- 统计面板独立为 `StatisticsPanel`

Vue 端 `feedback/list`、`member/list` 的内联弹窗参照 React 端已拆分的 `components/XxxDialog.tsx` 结构抽离。

### 4.2 Store slice 两端对齐

**现状**：两端 store 模块不对齐。

| Slice | Vue | React | 差异 |
|-------|:---:|:-----:|------|
| app/user/settings/permission/task/imageShow | ✓ | ✓ | 一致 |
| algorithm | ✓ | ✗ | React 缺失 |
| notification | ✓ | ✗(不补) | React 未读数在 MessageIcon 组件（单一消费者，不抽 slice） |
| tagsView | ✓ | ✗ | React 缺失（settingsSlice 有开关但无组件） |
| favorite | ✗ | ✓(broken) | React 有 slice 但 toggle 不调 API（见 §2.1），待修 |
| dataset | ✗ | ✓(死代码) | React 应删除（见 §2.2） |

**改造方向**：
- `notification` 未读数仅 `MessageIcon` 单一消费者，不抽 slice（违反"禁止过度设计"规则），保留组件局部 `useState` 即可
- `algorithm`/`tagsView` 是否补齐取决于功能需求：若 React 端确无算法全局状态需求可不补；TagsView 若要启用则需补 slice + 组件（见 §5.2）
- React `favorite` slice 需修复 toggle 调用 API（见 §2.1），非新增 slice

### 4.3 核心模块测试补充

**现状**：两端核心逻辑零测试覆盖。

| 模块 | Vue | React |
|------|:---:|:-----:|
| 权限守卫/动态路由生成 | ✗ | ✗ |
| task 轮询状态机 | ✗ | ✗ |
| favorite 逻辑 | 无 store | ✗（且有 bug） |
| 列表页业务逻辑 | ✗ | ✗ |

**改造方向**：优先补权限守卫、动态路由生成、task 轮询、favoriteSlice toggle 逻辑的单元测试。

---

## 五、P3：零碎技术债务

### 5.1 React imageShowSlice 持久化临时状态

**现状**：`imageShowSlice` 的 `whitelist` 包含 `loading`、`mouse`（鼠标坐标）、`mask`（遮罩坐标）、`width`/`height` 等纯临时状态。`loading` 持久化尤其危险——刷新时若处于 loading 状态，刷新后永远 loading。

**证据**：[imageShowSlice.ts:203-215](file:///e:/DehazeSystem/dehaze-front-react/src/store/modules/imageShowSlice.ts)

**改造方向**：whitelist 仅保留 `modelId`、`magnifier`、`divider`，移除 `loading`/`mouse`/`mask`/`width`/`height`/`naturalWidth`/`naturalHeight`/`urls`。

### 5.2 React 缺 TagsView 组件与动态 Breadcrumb

**现状**：
- `settingsSlice` 有 `tagsView` 开关，Settings 面板有"开启页面标签"选项，但无实际 TagsView 组件；其中"固定页面标签"Switch 仅为空壳（`{/* TODO */}`，无 `onChange`），属误导性死 UI
- [NavBar/index.tsx:70](file:///e:/DehazeSystem/dehaze-front-react/src/layout/components/NavBar/index.tsx) Breadcrumb 硬编码 `items={[{ title: "首页" }]}`，Vue 端有动态 Breadcrumb

**证据**：[Settings.tsx:129-131](file:///e:/DehazeSystem/dehaze-front-react/src/layout/components/NavBar/Settings.tsx)

**改造方向**：若启用 TagsView 则补组件 + slice；Breadcrumb 改为根据当前路由动态生成；一并移除"固定页面标签"空壳 Switch。若产品层面确认不需要 TagsView，则移除 Settings 面板的开关选项避免误导。

### 5.3 Vue task store DOM 操作外移

**现状**：`task` store 的 `downloadResult` 方法内含 `document.createElement('a')` 等 DOM 操作。

**证据**：[task.ts:49-56](file:///e:/DehazeSystem/dehaze-front-vue/src/store/modules/task.ts)

**改造方向**：将下载逻辑移至组件或独立工具函数（`useImportExport.ts` 已有 `downloadBlob` 可复用），store 仅返回下载 URL 与状态校验。

### 5.4 Vue permission 插件重复分支

**现状**：`setupPermission` 中"生成动态路由 + addRoute + 追加 404 兜底 + 置标志位 + next(replace)"的逻辑在两个分支中完全重复。

**证据**：[permission.ts](file:///e:/DehazeSystem/dehaze-front-vue/src/plugins/permission.ts) 第 33-56 行与第 60-78 行

**改造方向**：抽取 `addDynamicRoutes(roles)` 函数统一调用，消除分支差异。

---

## 六、不纳入改造的项

以下问题经评估不单列改造项，说明如下：

| 问题 | 不纳入原因 |
|------|-----------|
| store 导出风格不统一（`useXxxStore` vs `useXxxStoreHook`） | 纯风格偏好，无功能差异，在相关重构中顺带统一即可 |
| `any` 类型清理 | 普遍存在于 59 文件且非架构问题，应在日常开发中渐进消除，不宜单独立项 |
| 搜索防抖方式不统一 | 影响低，且多数页面用回车搜索无需防抖；统一为"回车搜索 + 清空重置"即可，属日常调整 |
| Electron `sandbox: false` | 当前仅加载本地内容无不可信来源，安全风险可控；扩展 IPC 功能时再评估 |
| `build.productName` 含 vue 字样 | trivial 配置错误，直接修改即可，无需列为改造项 |
| React Dialog 的 `forwardRef + useImperativeHandle` 模式 | 模式本身合理，与 Vue 声明式模式的差异属框架特性，无需强求统一 |
| 列表页数据是否进 store | 列表页局部状态用组件状态管理是合理架构选择，强行进 store 属过度设计 |
| Vue 静态 hidden 路由动态化（原 #9） | 静态 hidden 路由更简单可靠，强行动态化依赖后端菜单下发，属"为一致性而一致性"；仅在后端菜单接口已下发这些路由时顺带迁移，不单独立项 |
| React 推荐功能抽 useRecommendation hook（原 #11） | 实测 `analyze`/`getAlgorithmRecommendations` 仅 RecommendationWidget、algorithm-select 2 处重复（`recommendation/rules` 用 `getRules`/`getOption`，非分析流程）；2 处重复抽 hook 属可接受范围，价值不高，在相关重构中顺带处理 |
| 虚拟滚动（原 #12） | 无数据量支撑属投机优化，先用真实场景验证数据集详情页大列表是否真有性能问题再实施，当前不立项 |

---

## 七、改造优先级与建议顺序

1. **立即处理（P0）**：修复 React favoriteSlice toggle bug；批量删除两端死代码。这两项成本低、收益明确。
2. **短期推进（P1）**：dehazeParams 数据源统一、imageShow setter 清理、CRUD 样板抽象启用（止于现有 hook 组合）、错误处理去重、React i18n（前置：确认产品需求）。涉及核心数据流与状态管理，建议逐模块推进。
3. **中期规划（P2）**：巨型组件拆分、store slice 对齐、测试补充。改动面大，建议结合功能迭代逐步进行。
4. **择机处理（P3）**：imageShowSlice 持久化、TagsView/Breadcrumb（含移除空壳开关）、task store DOM 外移、permission 重复分支。在触及相关代码时顺带修复。
