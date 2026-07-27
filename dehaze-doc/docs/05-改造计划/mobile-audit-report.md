# 移动端项目待修复问题清单

> 本文档为移动端审计报告中**未修复问题**的追踪清单，已修复问题已删除。审计范围：dehaze-uniapp / dehaze-taro / dehaze_flutter / dehaze-react-native / dehaze-android。

---

## 一、功能缺失

### 1.1 Flutter 注册功能缺失

| 项目 | 内容 |
|------|------|
| 位置 | `e:\DehazeSystem\dehaze_flutter` |
| 文档要求 | 认证模块要求 Login/Register |
| 当前实现 | `auth_service.dart` L27 已定义 `register` 方法，但 `router/config.dart` 与 `pages/` 下均无 register 页面或路由 |
| 对比 | 其他端均已实现：Android（RegisterFragment）、UniApp（`src/pages/register/index.vue`）、Taro（`src/pages/register/index.tsx`）、React Native（`src/pages/register/index.tsx`） |

**修复方案**：在 Flutter 端新增 `RegisterPage` 并在 `router/config.dart` 注册路由，调用已有的 `auth_service.register()` 方法。

### 1.2 智能推荐未接入（UniApp/Taro/Flutter）

| 项目 | 内容 |
|------|------|
| 文档要求 | 算法选择模块：智能推荐 API |
| 当前实现 | UniApp/Taro/Flutter 三端 API 已定义但 UI 未调用；Android 和 React Native 已实现 |

**修复方案**：在三端的算法选择页面接入 `recommendAlgorithms` API，展示推荐算法列表。

---

## 二、代码质量

### 2.2 Android `DehazeApplication` 默认 BASE_URL 仅模拟器可用

| 项目 | 内容 |
|------|------|
| 位置 | `e:\DehazeSystem\dehaze-android\app\build.gradle` L30 `buildConfigField "String", "BASE_URL", '"http://10.0.2.2:8989"'` |
| 现状 | 已通过 `BuildConfig.BASE_URL` 解耦，不再在 Application 中硬编码（已修复），但默认值 `10.0.2.2:8989` 仅模拟器可用 |
| 影响 | 真机调试需手动修改 build.gradle 或注入不同 BASE_URL |

**修复方案**：在 `build.gradle` 中按 buildType 区分默认值（如 release 用生产域名），或通过 `local.properties` 注入开发者本地 IP。

### 2.3 各端对比页空状态重复

| 项目 | 内容 |
|------|------|
| 位置 | Taro/Flutter/RN/UniApp 的 5 个对比页面 |
| 问题 | 复制相同的"请先完成去雾处理"空状态组件 |

**修复方案**：各端提取 `CompareEmptyState` 共享组件。

### 2.4 各端 `formatFileSize` 重复实现

| 项目 | 内容 |
|------|------|
| 位置 | Taro（4 处）、Flutter（dataset 和 image_preview，且精度不一致：1位 vs 2位小数）、UniApp |

**修复方案**：各端提取为统一工具函数，统一精度（建议 2 位小数）。

### 2.5 各端死代码清理

| 端 | 死代码 |
|----|--------|
| Taro | `src/router/index.tsx`（整个文件为死代码，Taro 用 `app.config.ts`） |
| UniApp | `src/api/` 下 5 个未调用函数、`src/composables/useLayout.ts` 中 6 个未消费计算属性、`useUserStore.displayMode` |
| Flutter | `PaginatedSampleResponse`、`PageRequest`、`responsive_utils.dart` 中 6 个未调用工具（`getWaterfallColumnCount`、`getResponsiveFontSize`、`getCardAspectRatio`、`getResponsiveValue`、`ResponsiveBuilder`、`AdaptiveGridView`）、`MenuItemData.isNew`、`AppTheme.getShadow` 的 `isDark` 参数 |
| RN | 5 个死导出（`SIDE_NAV_WIDTH`、`ViewMode`、`predictBatch`、`useResponsiveValue`、`createResponsiveStyles`） |

### 2.6 各端 console 残留

| 端 | 残留情况 |
|----|---------|
| Taro | `dataset/index.tsx:151` 残留 `console.log("Image clicked:", ...)` |
| RN | 7 处 `console.warn` 残留 |
| UniApp | 19 处 console 语句（含 App.vue 3 处纯调试 log） |

**修复方案**：清理或替换为正式日志（timber/日志服务）。

---

## 三、UI/UX 优化

### 3.1 RN 滑块不支持拖拽

| 项目 | 内容 |
|------|------|
| 位置 | `dehaze-react-native` 的 ParamsPanel 和 Filter |
| 问题 | 滑块仅 +/- 按钮和点击轨道，不支持拖拽手势，参数调节体验差 |

**修复方案**：提取 `SliderControl` 组件，增加拖拽手势支持。

### 3.2 Flutter 计时器全局重建

| 项目 | 内容 |
|------|------|
| 位置 | `dehaze_flutter` 的 ProcessingPage |
| 问题 | 计时器每 100ms `setState` 一次，重建整棵 Widget 树 |

**修复方案**：计时器改为独立 `StatefulWidget`，仅重建自身。

### 3.3 UniApp 对比页返回栈过深

| 项目 | 内容 |
|------|------|
| 位置 | `dehaze-uniapp` 对比页面 |
| 问题 | 对比页面间用 `navigateTo` 互跳，用户探索 5 种模式后返回栈深达 5+ 层 |

**修复方案**：改为 `redirectTo` 或 Tab 容器，避免返回栈过深。

### 3.4 Android 缺少空状态/加载态

| 项目 | 内容 |
|------|------|
| 位置 | `dehaze-android` 的 CompareActivity/PresentationActivity/EvaluationActivity/ParallelFragment/OverlapFragment |
| 问题 | 无空状态占位，无图片时显示空白 |

**修复方案**：补充空状态/加载态占位组件。
