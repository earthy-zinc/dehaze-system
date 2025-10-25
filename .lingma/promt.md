# dehaze-front-react

我需要用 React + TypeScript 对 dehaze-front-vue 项目进行功能对齐式重构，目标是将 Vue 版本中已实现的全部业务功能、交互逻辑与用户体验完整迁移到 dehaze-front-react 项目中。

虽然 dehaze-front-react 已初步搭建了基础架构并实现了部分功能，但可能存在以下问题：

功能缺失（如缺少大部分页面和组件）
逻辑不一致（如权限校验、图片上传流程、对比交互行为）
未充分利用 React 生态特性（如 Redux Toolkit 状态管理、Ant Design 组件规范）
现要求按业务功能模块（如 dataset、evaluation、compare、user 等）逐模块进行审查与完善。每个模块必须严格按以下四步执行，每完成一步需经我审核确认后，方可进入下一步：

第一步：功能完整性比对
查阅 dehaze-front-vue 中当前模块的以下内容：
Pages（views）：页面路由、布局结构、核心交互流程
Components：业务组件（如 Waterfall、Magnifier、DraggableLine）及基础组件复用逻辑
Store（Pinia modules）：状态定义、actions、getters、持久化逻辑（如 localStorage 同步）
API 层：接口调用方式、错误处理、拦截器逻辑（如 Token 刷新、权限拦截）
特殊能力：WebSocket 实时进度、图片 MD5 校验、懒加载策略、CSS clip-path 对比等
对照 dehaze-front-react 中同名模块，检查是否存在：
页面缺失或路由未注册
组件未实现或交互行为不一致（如拖拽线逻辑、放大镜坐标计算）
状态管理未对齐（如缺少对应 Redux slice 或 action）
功能逻辑遗漏（如未实现图片并发上传、未集成 WebSocket）
输出：缺失/差异清单（按功能点分类）

第二步：核心逻辑与交互一致性校验
重点对比以下逻辑是否一致：
权限控制流程（角色-菜单-接口三级校验）
图片上传与校验逻辑（大小/格式/MD5/并发/进度条）
图像对比交互（clip-path + 拖拽分隔线 + 亮度对比度调节）
实时进度推送（WebSocket 连接、重连、消息解析）
状态持久化（Token、主题、布局模式是否存入 localStorage 并恢复）
若 React 实现与 Vue 行为不一致（即使“看起来能用”），需以 Vue 当前行为为准进行修正。
输出：逻辑差异说明 + 并直接编写代码进行新增/修正

第三步：代码质量与架构合规性检查
检查 dehaze-front-react 当前模块是否符合项目规范：
✅ 使用 Redux Toolkit（createSlice）管理状态，而非手写 reducer
✅ 使用 Ant Design 5.x 组件（如 Table、Upload、Slider），避免自造轮子
✅ 路由使用 React Router v6（createBrowserRouter 或 useRoutes）
✅ 组件为 函数式 + Hooks，无 class 组件
✅ 复杂逻辑抽离为 自定义 Hooks（如 useWebSocket、useImageUpload）
✅ 样式使用 UnoCSS 原子类 或 Ant Design 内置样式，避免全局 CSS 污染
✅ 错误边界、加载状态、空状态等用户体验完整
输出：架构/代码问题清单 + 直接修复代码

第四步：测试覆盖与行为验证
检查当前模块是否具备完整测试（使用 Vitest 或 Jest + React Testing Library）：
✅ 组件渲染测试（快照 + props 变化）
✅ 用户交互测试（拖拽、点击、表单输入）
✅ 状态流测试（Redux action → state → UI 更新）
✅ 异常路径（网络错误、WebSocket 断开、图片加载失败）
✅ 边界场景（空数据集、超大图、低网速）
要求：行覆盖率 ≥80%，关键路径 100% 覆盖。
若测试不足，需编写缺失测试用例并运行通过。请基于 dehaze-front-react 项目的实际代码结构和制定的《dehaze-front-react 项目测试规则[react-test-rule.md](./rules/react-test-rule.md)》
输出：测试覆盖率报告 + 新增测试代码

📌 执行规则（强化版）
每次仅处理一个业务模块
每完成上述四步中的任意一步，必须暂停并输出结构化报告，等待我审核确认。
所有输出需包含具体文件路径、代码片段（如必要）、行为描述，避免模糊表述。
若发现 Vue 原版存在明显缺陷，请提出，由我来确认如何去修正。


compare（图像对比：含 DraggableLine + clip-path + Magnifier）
dataset（数据集管理：含 Waterfall + 懒加载 + MD5 校验）
user（用户系统：含权限菜单 + Pinia 持久化）
evaluation（算法评估：含 ECharts 可视化）

# dehaze-uniapp

将 dehaze-front-vue（桌面端）业务功能重写至 dehaze-uniapp（多端小程序 + 移动 H5）

🎯 总体目标
完整迁移 dehaze-front-vue 的核心业务功能（用户系统、数据集管理、图像去雾、对比评估等）。
适配移动端交互逻辑：不照搬桌面端交互，而是基于 uView Plus 设计规范 与 移动端最佳实践 进行重构。
全平台兼容：确保代码在 微信小程序、支付宝小程序、H5（移动端） 上均可正常运行。
以功能实现为第一优先级：若某能力（如拖拽、Canvas）在小程序受限，则采用跨端通用方案 或 交互降级，但必须保证核心流程可用。

📦 执行方式：按业务模块逐个处理
每个模块必须严格按以下 四步流程 执行，每完成一步需经审核确认后，方可进入下一步。

⚠️ 模块划分依据：以 dehaze-front-vue 的业务功能为起点，但允许按移动端用户旅程拆分或合并页面（如将“上传+处理+对比”整合为单一流程页）。

第一步：功能映射与移动端交互适配分析
查阅 dehaze-front-vue 中当前模块的：
Pages（路由与布局）
Components（如 DraggableLine, Magnifier, Waterfall）
Store（Pinia 状态逻辑）
API 调用与 WebSocket 逻辑
分析哪些功能可直接迁移、哪些需交互重构、哪些需降级替代，例如：
拖拽分隔线对比 → 改为 滑动切换/双图左右滑动对比
Canvas 放大镜 → 改为 点击局部放大弹窗 或 双指缩放原图

输出：
✅ 移动端交互设计方案文档（含组件选型、手势逻辑、降级策略），撰写在 doc 文件夹中，每个模块一个文件

第二步：代码实现与架构合规性重构
按照 dehaze-uniapp 项目规范开发：
使用 Vue 3.4 + Composition API + <script setup>
UI 组件基于 uView Plus 3.x（禁用桌面端UI组件 Element Plus ）
状态管理：使用 Pinia（uni-app 支持） 或轻量级 reactive + storage 持久化
路由：使用 uni-app 原生页面路由（pages.json），非 Vue Router

确保代码结构符合 uni-app 最佳实践：
页面：/pages/模块名/页面名
组件：/components/业务组件
工具：/utils/（含兼容性封装，如 socket.ts, image.ts）
输出：
✅ 符合规范的代码实现
✅ 架构说明（文件组织、状态流、API 调用链），撰写在 doc 文件夹中，每个模块一个文件

第三步：跨平台兼容性验证
验证以下平台功能可用性：
✅ 微信小程序（基础库 ≥ 2.24.0）
✅ 支付宝小程序
✅ 移动端 H5（iOS Safari + Android Chrome）
重点验证：
图片上传/预览是否正常
WebSocket 是否稳定连接并接收消息
页面跳转、返回、Tab 切换是否符合小程序导航规范
内存占用是否过高（避免长列表未虚拟滚动）

第四步：测试覆盖与体验优化
编写测试（以手动测试为主，可辅以 H5 端单元测试）：
正常流程：用户登录 → 上传图片 → 选择算法 → 查看结果 → 对比评估
异常路径：网络中断、图片格式错误、服务端错误码
边界场景：空数据集、超大图（>10MB）、弱网环境
优化移动端体验：
加载状态（骨架屏 or u-loading）
操作反馈（u-toast 提示）
手势友好（避免小点击区域、支持返回手势）
性能保障（图片懒加载、列表虚拟滚动）
输出：
✅ 测试代码，所有测试代码必须通过 TypeScript 类型检查
不使用 any，充分利用项目中 typings/ 和 本地库 [dehaze-sdk-js](/dehaze-tool/dehaze-sdk-js/README.md) 

📌 执行规则
每次仅处理一个业务模块
每完成上述四步中的任意一步，必须暂停并提交结构化报告，等待您审核确认。
所有交互设计建议需附带移动端合理性说明（如“滑动切换比拖拽更适合单手操作”）。
若某功能在所有小程序平台均无法实现，需提出替代方案并请您决策。
逐步沉淀 dehaze-uniapp 自有设计规范（从第一个模块开始记录组件使用、交互模式、错误处理等）

# dehaze-taro
将 dehaze-front-vue（桌面端）完整重写为 dehaze-taro（多端小程序 + 移动 H5）

🎯 总体目标
功能对齐：完整迁移用户系统、数据集管理、图像去雾、对比评估等核心业务。
体验适配：不照搬桌面交互，而是基于 移动端最佳实践 重构（如拖拽 → 滑动切换，Canvas → 图片预览）。
架构现代化：采用 Redux Toolkit（RTK） 替代传统 Redux，提升可维护性。
全平台兼容：确保在 微信小程序、支付宝小程序、H5（移动端） 上功能可用、体验一致。
以实现为准：若某能力在小程序受限，采用 Taro 跨端通用方案 或 合理交互降级，但必须保障主流程畅通。
📦 执行方式：按业务模块逐个处理
每个模块必须严格按以下 四步流程 执行，每完成一步需经审核确认后，方可进入下一步。

⚠️ 模块划分以 dehaze-front-vue 业务功能为起点，但允许按移动端用户旅程重组页面结构（如合并“上传+处理+对比”为单一流程页）。

第一步：功能映射与移动端交互适配分析
分析 dehaze-front-vue 当前模块的：
Pages（路由、布局、交互流）
Components（如 DraggableLine, Magnifier, Waterfall）
Store（Pinia 状态逻辑）
API 与 WebSocket 逻辑
输出：
✅ 移动端交互设计方案（含 @taroify/core 组件选型）：
✅ 跨端能力适配策略

第二步：代码实现与架构现代化重构
技术规范：
框架：Taro 4.0 + React 18 + TypeScript
状态管理：Redux Toolkit（createSlice, configureStore）
UI：以 @taroify/core 为主（Swiper, ImagePreview, Slider, Button 等）
网络层：封装 request.ts，统一处理 token、错误拦截、loading

输出：
✅ 符合 Taro 多端规范的代码
✅ RTK 状态流设计说明（UML图等，写在doc文件夹）
✅ 组件与页面结构图

第三步：多端兼容性验证
在以下平台验证：
✅ 微信小程序（基础库 ≥ 2.24.0）
✅ 支付宝小程序
✅ 移动端 H5（iOS/Android 主流浏览器）
验证重点：
页面跳转是否符合小程序导航规范
图片上传/预览是否正常
WebSocket 是否稳定接收进度消息
列表滚动、内存占用是否流畅

第四步：测试覆盖与体验优化
体验优化：
加载状态：@taroify/loading 或骨架屏
操作反馈：Taro.showToast
手势友好：避免小点击区域，支持返回手势
性能：图片懒加载、列表虚拟滚动（如需）
输出：
✅ 测试代码，所有测试代码必须通过 TypeScript 类型检查
不使用 any，充分利用项目中 typings/ 和 本地库 [dehaze-sdk-js](/dehaze-tool/dehaze-sdk-js/README.md) 

📌 执行规则（强制约束）
每次仅处理一个业务模块
每完成上述四步中的任意一步，必须暂停并提交结构化报告，等待您审核确认。
所有交互设计必须附带移动端合理性说明（如“滑动切换更适合单手操作”）。
若某功能在所有目标平台均无法实现，需提出替代方案并请您决策。
逐步沉淀 dehaze-taro 自有设计规范（从第一个模块开始记录组件使用、状态管理、错误处理等）。

# dehaze-react-native

将 dehaze-front-vue（桌面 Web）完整重写为 dehaze-react-native（iOS + Android 原生应用）

🎯 总体目标
功能对齐：完整迁移用户系统、数据集管理、图像去雾、对比评估、实时进度等核心业务。
体验合规：遵循 iOS Human Interface Guidelines 与 Android Material Design，不照搬 Web 交互。
架构现代化：采用 Redux Toolkit（RTK） 实现类型安全、模块化状态管理。
能力适配：充分利用 React Native 生态（如 VisionCamera、Reanimated、PagerView），对 Web 专属能力（如 clip-path、原生 File API）进行合理修改。
以实现为准：若某功能在移动端受限，采用 跨平台通用方案 或 体验等效替代，确保主流程畅通。
📦 执行方式：按业务模块逐个处理
每个模块必须严格按以下 四步流程 执行，每完成一步需经您审核确认后，方可进入下一步。

⚠️ 模块划分以 dehaze-front-vue 业务功能为起点，但允许按移动端用户旅程重组页面结构（如合并“上传 → 处理 → 对比”为单一流程页）。

第一步：功能映射与移动端交互适配分析
分析 dehaze-front-vue 当前模块的：
Pages（路由、布局、交互流）
Components（如 DraggableLine, Magnifier, Waterfall）
Store（Pinia 状态逻辑）
API 与 WebSocket 逻辑
输出：
✅ 移动端交互设计方案（含组件选型与手势逻辑）：
✅ 跨端能力适配策略

第二步：代码实现与架构现代化重构
技术规范：
框架：React 19.1 + React Native 0.81.4 + TypeScript
状态管理：Redux Toolkit（createSlice, configureStore, createAsyncThunk）
导航：@react-navigation（Stack + Bottom Tabs + Drawer 按需组合）
UI 组件：
基础：View, Text, Image, ScrollView, FlatList
交互：react-native-gesture-handler + react-native-reanimated
图像：react-native-image-picker, react-native-vision-camera
对比：react-native-pager-view

输出：
✅ 符合 React Native 最佳实践的代码
✅ RTK 状态流设计说明（含异步 thunk 与类型推导）
✅ 组件与页面结构图

第三步：测试覆盖与体验优化
测试范围：
正常流程：登录 → 选择图片/摄像头 → 处理 → 对比 → 查看指标
异常路径：网络中断、图片加载失败、权限拒绝、服务端错误
边界场景：空数据集、超大图（>10MB）、弱网、低电量模式
体验优化：
加载状态：ActivityIndicator + 骨架屏
操作反馈：Toast 或原生 Snackbar
手势友好：避免小点击区域（≥44pt），支持系统返回
性能：图片懒加载、列表虚拟化、避免不必要的 re-render
输出：
✅ 测试代码，所有测试代码必须通过 TypeScript 类型检查
不使用 any，充分利用项目中 typings/ 和 本地库 [dehaze-sdk-js](/dehaze-tool/dehaze-sdk-js/README.md) 

📌 执行规则（强制约束）
每次仅处理一个业务模块
每完成上述四步中的任意一步，必须暂停并提交结构化报告，等待您审核确认。
所有交互设计必须附带移动端合理性说明（如“滑动切换符合 iOS Page Control 模式”）。
若某功能在 iOS 或 Android 任一平台无法实现，需提出替代方案。
逐步沉淀 dehaze-react-native 自有设计规范（从第一个模块开始记录组件使用、状态管理、错误处理、权限申请流程等）。

# dehaze-android

将 dehaze-front-vue（桌面 Web）完整重写为 dehaze-android（原生 Android 应用，Java 语言）

🎯 总体目标
功能对齐：完整迁移用户认证、数据集管理、图像上传、去雾处理、结果对比、指标评估等核心业务。
体验合规：遵循 Android Material Design 3 与 Google 人机交互指南，重构桌面端交互（如拖拽 → 滑动切换，画布 → 图片预览）。
架构现代化：采用 Android Jetpack 架构组件（Navigation、Room、DataStore、ViewModel + LiveData）实现清晰分层。
能力适配：充分利用 Android 原生能力（如 CameraX、系统相册、通知、后台任务），对 Web 专属能力（如 WebSocket、Canvas、File API）采用 Android 通用方案或合理降级。
以实现为准：若某功能在 Android 上受限（如实时 Canvas 渲染），优先采用 系统级组件或体验等效替代，确保主流程可用。
📦 执行方式：按业务模块逐个处理
每个模块必须严格按以下 四步流程 执行，每完成一步需经您审核确认后，方可进入下一步。

⚠️ 模块划分以 dehaze-front-vue 业务功能为起点，但允许按 Android 用户旅程重组页面结构（如合并“上传+处理+对比”为单一流程 Activity/Fragment）。

第一步：功能映射与 Android 交互适配分析
分析 dehaze-front-vue 当前模块的：
Pages（路由、布局、交互流）
Components（如 DraggableLine, Magnifier, Waterfall）
Store（Pinia 状态逻辑）
API 与 WebSocket 逻辑
输出：
✅ 功能可实现性评估表
✅ Android 交互设计方案（含组件与系统能力选型）：

图像对比 → 使用 ViewPager2 + FragmentStateAdapter 实现左右滑动切换（符合 Material 模式）
放大镜细节查看 → 使用 Glide 加载高清图 + PhotoView（需引入 PhotoView）或系统 ImageView + 双击缩放
拖拽分隔线 → 降级为 Tab 切换或滑动对比（因 Android 无高效 clip-path 实现，且拖拽体验差）
瀑布流展示 → 使用 RecyclerView + StaggeredGridLayoutManager
实时进度推送 → 若需 WebSocket，使用 OkHttp 的 WebSocketListener；否则改用 轮询 + 通知栏进度提示
摄像头实时捕获 → 推荐集成 CameraX（虽未在依赖中，但为 Android 官方推荐）

✅ 跨能力适配策略：
图片选择 → 调用系统 Intent.ACTION_PICK 或 ActivityResultContracts.GetContent
文件上传 → Retrofit + MultipartBody.Part，MD5 校验交由后端处理
本地持久化 → 使用 DataStore<Preferences> 存储 token/设置，Room 存储数据集元数据
网络层 → Retrofit + OkHttp + Timber 日志，统一拦截器处理 token 注入

第二步：代码实现与 Android 架构合规重构
技术规范：
语言：Java
架构：MVVM + Repository 模式
View：Activity / Fragment（使用 Navigation Component 路由）
ViewModel：管理 UI 状态（LiveData）
Repository：协调 Room、Retrofit、DataStore
Data：Room Entity + DAO，Retrofit API 接口

UI 组件：
基础：Material Design 组件（MaterialButton, TextInputLayout, BottomNavigationView）
列表：RecyclerView + DiffUtil
图片：Glide + PhotoView（建议新增）
导航：Navigation Component（Fragment + NavGraph）
输出：
✅ 符合 Android 官方架构指南的 Java 代码
✅ 架构分层说明（数据流：UI → ViewModel → Repository → Data Source）
✅ 权限申请与生命周期处理方案（如相机、存储）

第三步：Android 多版本兼容性验证
验证重点：
权限请求是否合规（运行时权限）
图片加载是否内存泄漏（Glide 自动处理）
后台网络是否受限（Android 9+ 网络安全配置）
Navigation 返回栈是否符合用户预期
通知、进度提示是否可见

第四步：测试覆盖与体验优化
测试范围：
正常流程：登录 → 选图/拍照 → 上传 → 处理 → 对比 → 查看指标
异常路径：无网络、权限拒绝、图片加载失败、服务端错误
边界场景：大图（>10MB）、弱网、低存储空间
体验优化：
加载状态：ProgressBar + 占位图
操作反馈：Snackbar 或 Toast
手势友好：按钮 ≥ 48dp，支持系统返回键
性能：图片缩略图预加载、RecyclerView 池复用
输出：
✅ 测试代码，充分利用本地库 [dehaze-sdk-android](/dehaze-tool/dehaze-sdk-android/README.md) 

📌 执行规则（强制约束）
每次仅处理一个业务模块
每完成上述四步中的任意一步，必须暂停并提交结构化报告，等待您审核确认。
所有交互设计必须附带 Android 合理性说明（如“ViewPager2 符合 Material Design 滑动切换模式”）。
若某功能在 Android 上无法实现（如高性能 Canvas 实时渲染），需提出替代方案并请您决策。
逐步沉淀 dehaze-android 自有设计规范（从第一个模块开始记录组件使用、权限流程、错误提示文案、加载策略等）。

# dehaze-front-vue

你是一位资深前端测试工程师，熟悉 Vue 3 + TypeScript + Vite + Pinia + Vitest + Playwright 技术栈，并严格遵循工程规范。

请基于 dehaze-front-vue 项目的实际代码结构[front-vue-rule.md](./rules/front-vue-rule.md)和制定的《dehaze-front-vue 项目测试规则（[vue-test-rule.md](./rules/vue-test-rule.md)》）》，为项目的各个模块编写可直接集成、高覆盖率、符合规范的测试代码：

你必须遵守 vue-test-rule.md 测试规则的约束，覆盖率要求：≥ 80%（行、函数、分支、语句）
测试不依赖实现细节，只验证行为与输出
每个 it 只测一个功能点
覆盖成功、失败、边界、异常路径

请为每个模块提供完整、可运行、带注释的测试代码

所有测试代码必须通过 TypeScript 类型检查
不使用 any，充分利用项目中 typings/ 和 本地库 [dehaze-sdk-js](/dehaze-tool/dehaze-sdk-js/README.md) 

测试名称简洁明了，说明测试目的

目标：生成的测试代码应立即可集成到项目中，助力达成 单元测试覆盖率 ≥ 80%、Lighthouse 性能 > 92 的质量目标。