# 移动端项目审计报告

审计范围：dehaze-uniapp / dehaze-taro / dehaze_flutter / dehaze-react-native / dehaze-android
对照文档：dehaze-doc/docs（产品设计、模块设计、架构文档、UI/UX规范）
审计日期：2026-07-22

---

## 一、功能完整性总览

| 功能模块 | Taro | Flutter | RN | Android | UniApp |
|---------|------|---------|-----|---------|--------|
| 登录(含验证码) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 注册 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 图像输入(上传/相机/样例/历史) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 算法选择(树形/搜索) | ✅ | ✅ | ✅ | ✅ | ⚠️ 仅平铺列表 |
| 智能推荐 | ❌ | ❌ | ✅ | ✅ | ❌ API已定义未调用 |
| 去雾处理(参数/进度/结果) | ✅ | ✅ | ✅ | ✅ | ⚠️ 无实时进度 |
| 效果对比(6种模式) | ✅ | ✅ | ✅ | ⚠️ 仅2种 | ✅ |
| 数据集管理 | ✅ | ✅ | ✅ | ✅ | ⚠️ 无CRUD |
| 任务历史 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 个人中心 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 系统管理 | ✅ | ❌ | ❌ | ✅ | ❌ |
| Dashboard | ✅ | ❌ | ❌ | ✅ | ❌ |

**共性问题**：所有端均未实现注册功能（文档要求 Login/Register）。

---

## 二、各端关键问题

### 2.1 dehaze-uniapp（问题最多）

**功能缺失：**
- 智能推荐 API (`recommendAlgorithms`) 已定义但从未调用
- 算法树形结构未渲染（`Algorithm.children` 字段存在但 UI 为平铺列表）
- 处理进度轮询 (`getPredictionStatus`) 未实现，仅有 spinner
- 数据集 CRUD 接口已定义但无 UI 入口
- 系统管理模块完全缺失

**代码质量：**
- `getImageInfo` 辅助函数在 4 个组件中完全重复（CameraArea/QuickStartCard/UploadArea/SampleGallery）
- 页面头部卡片 `.page-header-card` 在 10+ 页面中复制粘贴，应抽为组件
- `src/api/` 下 5 个函数定义后从未被调用（`getAlgorithmOptions`、`recommendAlgorithms`、`getPredictionStatus`、`getEvalTaskStatus`、数据集CRUD系列）
- `src/composables/useLayout.ts` 中 6 个响应式计算属性从未被消费
- `useUserStore` 在 App.vue 初始化但 `displayMode` 从未被使用
- `src/api/auth.ts` 4 个函数为纯透传包装，零附加值
- 设计令牌系统（`variables.scss` + `common.scss`）完全未被使用，所有页面硬编码色值/间距

**潜在 Bug：**
- `home/index.vue` 用 `uni.navigateTo` 跳转 tabBar 页面（image-input、dataset），真机会静默失败，必须用 `switchTab`
- `pages.json` 声明了原生 tabBar + 自定义 `Tabbar.vue` 同时存在，H5 端可能渲染双底栏
- `Tabbar.vue:126` 的 `padding-bottom: calc(env(safe-area-inset-bottom) + 50px)` 在多数设备产生过大空白
- `side-by-side/index.vue` 中 `.header-title` 的 `color: #fff` 被后续规则覆盖，属死 CSS

**UI/UX：**
- 对比页面间用 `navigateTo` 互跳，用户探索 5 种模式后返回栈深达 5+ 层
- 算法搜索清除按钮仅 `padding: 8rpx`（约4px），远低于 44px 最小触控标准
- 任务历史操作按钮 `padding: 10rpx 20rpx` 同样过小
- 首页算法数量加载无 loading 态，显示 `0` 直到请求完成
- 19 处 console 语句残留（含 App.vue 3 处纯调试 log）

---

### 2.2 dehaze-taro（质量最好）

**功能缺失：**
- 登录页无注册入口

**代码质量：**
- `src/router/index.tsx` 整个文件为死代码（React.lazy 路由定义，Taro 用 `app.config.ts`）
- `formatFileSize` 在 4 处重复实现（ImageCard/ImageViewer/imageInput service/processing page）
- 首页 5 个组件对 `Taro.navigateTo` 包了 try/catch，但 navigateTo 走 fail 回调不抛异常，catch 永远不会触发
- 5 个对比页面在 useState 初始化器和 useEffect 中双重调用 `loadCompareContext()`
- 保存到相册逻辑在 `processing/index.tsx` 和 `CompareToolbar` 中重复实现
- `src/utils/permission.ts` 4 个函数 + `src/config/menu.ts` 6+ 个导出从未被调用
- `login/index.tsx:21` 空的 `useDidShow(async () => {})` 无意义
- `dataset/index.tsx:151` 残留 `console.log("Image clicked:", ...)`
- 混用 `.less` 和 `.scss` 无明确规则

---

### 2.3 dehaze_flutter

**功能缺失：**
- 系统管理、Dashboard 未实现（Tier 3 定位可接受）
- 对比页快捷操作（保存/分享/导出/收藏）均为 "功能开发中" toast

**代码质量：**
- `_buildNoData` 组件在 5 个对比页面中复制粘贴，应提取为共享 Widget
- `_formatFileSize` 在 dataset 和 image_preview 中重复（且精度不一致：1位 vs 2位小数）
- `_showSnackBar`/`_showError` 在 4 个页面中重复
- `PaginatedSampleResponse` 模型定义后从未使用
- `PageRequest` 类定义后从未使用
- `responsive_utils.dart` 中 6 个工具函数/Widget 从未被调用（`getWaterfallColumnCount`、`getResponsiveFontSize`、`getCardAspectRatio`、`getResponsiveValue`、`ResponsiveBuilder`、`AdaptiveGridView`）
- `MenuItemData.isNew` 字段声明后从未读取
- `AppTheme.getShadow` 接受 `isDark` 参数但从未使用
- `DatasetService` 手动检查 `code == successCode`，与 ResponseInterceptor 逻辑重复
- `ImageInputService.fetchSamples` 绕过拦截器模式，手动检查响应码
- `api_config.dart` 顶部 `import 'dart:io'` 在 Web 编译时会报错（项目含 web/ 目录）
- `FileService` 手动设置 `Content-Type: multipart/form-data`（Dio 自动处理）

**UI/UX：**
- ProcessingPage 计时器每 100ms setState 一次，重建整棵 Widget 树
- 对比页底部模式切换栏在小屏无滚动容器，可能溢出
- 4 个对比页传空字符串 `subtitle: ''` 给 ComparisonScaffold，渲染空 Text Widget
- 数据集深度链接无 loading 态
- 任务历史无下拉刷新

---

### 2.4 dehaze-react-native

**功能缺失：**
- 算法详情页用户评价、相关链接、效果样例为静态占位符
- 用户资料仅只读，无编辑功能

**代码质量：**
- 7 处 `console.warn` 残留
- 5 个对比页面重复相同的空状态组件 + 样式（应提取 `CompareEmptyState`）
- 图片尺寸 fallback 逻辑在 UploadArea 和 CameraCapture 中重复
- Slider/Stepper UI 在 ParamsPanel 和 Filter 中重复（应提取 `SliderControl`）
- 控制栏样式在 Overlay/Magnifier/Filter 中重复
- `home/index.tsx:61-65` 五个变量别名指向同一函数
- 死导出：`SIDE_NAV_WIDTH`、`ViewMode`、`predictBatch`、`useResponsiveValue`、`createResponsiveStyles`
- 主题令牌冗余：`success/warning/error` 与 `status.*` 重复；`sizes.caption == sizes.small`、`sizes.body == sizes.medium`
- Task 页和 Dataset 组件大量硬编码色值，未使用 theme tokens

**潜在 Bug：**
- `SideBySide.tsx:143` 提示"双击图片可放大查看"，实际实现为单击
- 4 个对比页面在模块顶层调用 `Dimensions.get('window')`，旋转屏幕不会更新尺寸

**UI/UX：**
- 滑块不支持拖拽（仅 +/- 按钮和点击轨道），参数调节体验差
- Profile 页 `refreshUserInfo()` 失败静默吞掉

---

### 2.5 dehaze-android（问题最严重）

**功能性 Bug：**
- `PresentationActivity.ResultPagerAdapter` 调用 `notifyItemChanged()` 后 FragmentStateAdapter 不会重建已存在的 Fragment，导致 ViewPager 中图片 URL 更新后**界面不刷新**
- `AlgorithmViewModel.java` 使用 `RepositoryCallback` 但未 import，存在编译风险
- `CompareViewModel.predictMultiple()` 用 `int[] pending` 计数器在多线程回调中递减，存在竞态条件（应用 `AtomicInteger`）

**代码质量（严重）：**
- `safe(String)` 方法在 10+ 个文件中重复定义，尽管 `StringUtils.safe()` 已存在
- `safeParseLong(String)` 在 3 个 Activity 中重复
- `getFileNameFromUri(Uri)` 15 行方法在 3 个 Activity 中完全相同
- `copyToCache(Uri)` 20 行方法在 2 个 Activity 中完全相同
- `updateAlgorithmSpinner(List<Option>)` 在 3 个 Activity 中完全相同
- `BaseViewModel` 已提供 loading/error/operationResult，但仅 2/14 个 ViewModel 继承，其余 12 个手动重复声明
- `build.gradle` 声明 Room/DataStore/CameraX 依赖但项目中完全未使用（增加 APK 体积）
- `compileSdk 36` 不存在（最新稳定版为 35）
- `DehazeApplication` 硬编码 `http://10.0.2.2:8989`（仅模拟器可用），无环境配置

**UI/UX：**
- LoginFragment 有 loading LiveData 但无 ProgressBar 或按钮禁用
- CompareActivity/PresentationActivity/EvaluationActivity 无空状态占位
- ParallelFragment/OverlapFragment 无图片时仅显示空白 ImageView
- 混用 ViewBinding 和 findViewById
- 混用 `androidx.appcompat.app.AlertDialog` 和 `android.app.AlertDialog`
- LoginFragment 用原生 `Toast.makeText()`，其余用 `ToastUtils.showShort()`
- `DatasetDetailFragment:167` 使用已废弃的 `onBackPressed()`
- 无 ViewModel 在 `onCleared()` 中取消网络请求

---

## 三、跨端共性问题

| 问题类型 | 具体表现 | 涉及端 |
|---------|---------|--------|
| 对比页空状态重复 | 5 个对比页面复制相同的"请先完成去雾处理"空状态 | Taro/Flutter/RN/UniApp |
| formatFileSize 重复 | 同一工具函数在多处独立实现 | Taro/Flutter/UniApp |
| 注册功能缺失 | 文档要求 Login/Register，所有端仅实现 Login | 全部 |
| 智能推荐未接入 | API 已定义但 UI 未调用 | UniApp/Taro/Flutter |
| console 残留 | 生产代码中残留调试日志 | Taro/RN/UniApp |
| 设计令牌未贯彻 | 定义了 token 系统但页面硬编码色值 | UniApp/RN |
| 死代码/未使用导出 | 定义后从未调用的函数/类/变量 | 全部 |

---

## 四、优先级建议

### P0 - 功能性 Bug（立即修复）
1. Android `PresentationActivity` ViewPager 不刷新
2. Android `AlgorithmViewModel` 缺少 import（编译风险）
3. Android `CompareViewModel` 线程安全
4. UniApp `navigateTo` 跳转 tabBar 页面真机失败
5. UniApp 双 TabBar 渲染

### P1 - 代码重复/架构（短期重构）
1. Android 提取公共工具方法，消除 10+ 处 `safe()` 重复
2. Android 统一 BaseViewModel 继承
3. 各端提取对比页空状态为共享组件
4. 各端提取 `formatFileSize` 为统一工具
5. UniApp 提取页面头部卡片组件
6. RN 提取 SliderControl 组件

### P2 - 死代码清理（日常维护）
1. Taro 删除 `src/router/index.tsx`
2. UniApp 删除未调用的 API 函数和 useLayout 死代码
3. Flutter 删除 `PaginatedSampleResponse`、`PageRequest`、6 个未用响应式工具
4. RN 删除 5 个死导出
5. Android 移除未使用的 Room/DataStore/CameraX 依赖
6. 各端清理 console 语句

### P3 - 体验优化（迭代改进）
1. UniApp 对比页改为 redirectTo 或 Tab 容器（避免返回栈过深）
2. RN 滑块增加拖拽手势
3. Flutter 计时器改为独立 Widget 避免全局重建
4. Android 补充空状态/加载态
5. 各端补充注册功能
6. UniApp 接入智能推荐和算法树形展示
