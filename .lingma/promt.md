# dehaze-front-react

我需要用 React + TypeScript 对 dehaze-front-vue 项目进行功能对齐式重构，目标是将 Vue 版本中已实现的全部业务功能、交互逻辑与用户体验完整迁移到 dehaze-front-react 项目中.

虽然 dehaze-front-react 已初步搭建了基础架构并实现了部分功能，具体请参阅 dehaze-front-vue 和 dehaze-front-react 的规则文件，但可能存在以下问题：

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

第二步：代码实现与架构现代化重构
请思考代码整体架构如何实现，编写代码详细设计文档，给出方案，随后编写代码实现
框架：Taro 4.0 + React 18 + TypeScript
UI：以 @taroify/core 为主
网络层：基于 [dehaze-sdk-js](/dehaze-tool/dehaze-sdk-js/README.md) 封装的API

第三步：多端验证
编译和运行微信小程序、支付宝小程序、移动端 H5（iOS/Android 主流浏览器）来验证代码是否正确，并分析代码架构是否

📌 执行规则（强制约束）
每次仅处理一个业务模块
每完成上述三步中的任意一步，必须给出结构化报告文档。
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
输出（写入doc文件夹中）：
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
✅ 直接新建/编辑代码，给出符合 Android 官方架构指南的 Java 代码，
✅ 架构分层说明（数据流：UI → ViewModel → Repository → Data Source）

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

# 需求推导

你是一个资深的软件架构师和需求分析师，具备从代码反向推导业务需求与系统设计的丰富经验。

你的任务是：根据我提供的单个代码文件或代码片段，深入分析其实现逻辑、调用关系、数据结构和注释信息，逆向推理出该代码所对应的具体功能需求，并生成一份结构清晰、内容完整的需求与设计文档。

请严格按照以下要求执行：

📌 任务目标
基于对应功能前后端代码，推导并撰写该功能模块的独立需求文档到 [图像去雾系统](dehaze-doc\docs\项目文档\图像去雾系统) 文件夹下，仅聚焦当前代码所实现的功能点，不涉及其他模块。

📂 输入内容
包括但不限于我提供的代码文件、文件路径/模块名称

🔍 分析步骤（请按此逻辑思考）
功能定位：判断该代码属于哪个业务模块（如用户管理、订单处理、支付网关等），实现的是什么具体功能。
需求还原：从代码行为反推原始业务需求——用户或系统“为什么要实现这个功能”？解决什么问题？达到什么目标？
接口与交互分析：识别输入输出参数、API 路由、数据库操作、外部服务调用等，还原系统交互设计。
非功能性需求推断：根据代码实现方式（如加锁、缓存、校验逻辑等），推测性能、安全性、可靠性等方面的要求。
设计思路归纳：总结代码中体现的设计模式、分层结构、异常处理机制等。

输出格式参考：
```markdown
# 模块名称：[填写模块/功能名称]

## 1. 功能概述
简要描述该模块的核心作用和业务价值。

## 2. 原始需求背景
说明该功能产生的业务场景和用户痛点，还原需求提出的原因。

## 3. 需求详情
- **功能需求**：列出具体的功能点（如“支持邮箱登录”“校验密码强度”）
- **非功能需求**：
  - 性能要求
  - 安全要求
  - 可用性要求
  - 兼容性要求

## 4. 核心流程
用文字描述关键处理流程，例如：
1. 用户提交登录请求
2. 系统验证邮箱格式
3. 查询用户是否存在
4. 校验密码哈希值
5. 生成 JWT Token 返回

## 5. 设计说明
解释代码中体现的设计思想，例如：
- 使用了哪种设计模式？
- 为何采用当前的分层结构？
- 是否考虑了可扩展性或并发安全？

## 9. 待确认事项
列出需要进一步澄清的问题（如有模糊逻辑、缺少上下文等情况）。
```

每次只分析一个功能点，输出一份文档，不要试图覆盖整个系统。
输出语言为中文，保持专业、简洁、可读性强。

# 模块说明文档

你是一名资深软件架构师兼技术文档工程师。请基于当前模块 完整 Java 源代码，改造或生成一份

你必须做到：
1. 深度阅读并理解全部代码，包括类结构、方法逻辑、注释、配置、依赖、测试用例。
2. 所有结论必须有代码依据，禁止无根据的推测。
3. 优先使用代码中的命名、注释和结构作为文档内容来源。
4. 文档需面向多角色读者：开发、测试、运维、产品经理，语言需准确且具备技术深度。
5. 输出格式为标准 Markdown，支持 Mermaid 图表（架构图、时序图等）。

---

输出文档结构与内容规范

请按照以下章节组织输出，每部分需基于代码分析得出：
1. 总体说明
核心职责：总结模块解决的核心问题，基于主类、Service 接口、Controller 路径归纳。
边界声明：通过分析依赖注入、接口暴露、包划分，明确“本模块负责什么，不负责什么”。
2. 需求与背景
业务动机：从类/方法注释、测试用例描述、配置项语义中推断业务场景。
技术驱动因素：分析是否引入新框架（如从 synchronized 改为 Redisson）、是否解决性能/安全问题
需求映射：将代码功能反向映射到用户故事
3. 功能与非功能需求分析
功能性需求、非功能性需求：性能、安全、可观测性、可靠性
4. 技术栈与依赖解析
从 pom.xml / build.gradle 提取 Spring Boot、JDK、关键库版本，分析核心依赖和其用途，用到了哪些功能，涉及到的技术详细介绍
5. 架构设计
5.1 分层结构
分析包结构、识别核心领域模型（Entity/Aggregate）
5.2 组件交互图（Mermaid）
图中组件名必须与代码类名一致（如 UserAuthController）。
5.3 关键流程时序图（Mermaid）
选择多个核心业务流程，基于方法调用链绘制，反映真实调用顺序。
6. 核心实现详解/核心算法/逻辑说明
对复杂方法（如状态机、规则引擎、加密流程）进行逐步解释。

---
要求
准确、一致、无幻觉、专业避免口语化，让任何工程师在不阅读代码的情况下，通过该文档即可理解模块设计意图、使用方式。
请将结果修改和写入到该模块的README.md中。
如要引用代码文件，请严格遵守`[]()`markdown语法并采用相对路径。


请利用Context7 MCP 获取 uview-plus 最新官方文档，将其中所有组件用法文档保存下来，记录为markdown格式，你需要严格保持内容和官方文档一致，保存在dehaze-uniapp\doc\uview-plus文件夹下，按照官方文档的分类新建子文件夹，如基础组件、表单组件、数据组件、布局组件等。每个组件一个文件，文件名与组件名一致。请先开始保存一个组件到对应文件夹下，等我审核后，再开始保存下一个组件。

请你来撰写一篇10万字面向已经精通Java Maven相关技术栈的中级后端开发人员的 gradle 从入门到精通的全套教程。请你先列出教程大纲，将大纲写在放在`E:\DehazeSystem\dehaze-doc\docs\gradle`根目录下，作为当前任务的记忆，每次写完一个章节都可以回顾一下该文档。

你可以在适当的时机采用 sequential thinking mcp 进行规划、context7 mcp 获取最新的知识。

请你来撰写一篇5万字面向已经精通Java相关技术栈的中级后端/安卓开发人员的 Groovy DSL语言 从入门到精通的全套教程。请你先列出教程大纲，将大纲写在放在 `E:\DehazeSystem\dehaze-doc\docs\前端开发\Groovy` 根目录下，作为当前任务的记忆，每次写完一个章节都可以回顾一下该文档。

你可以在适当的时机采用 sequential thinking mcp 进行规划、context7 mcp 获取最新的知识。

然后是具体的教程文档要求，采用markdown格式，分章节写入到目录 `E:\DehazeSystem\dehaze-doc\docs\前端开发\Groovy` 下，在该文件夹下每个部分创建一个文件夹，在每个部分内部，每个章节为一个md文件，文件名采用章节名。内容你需要分多个步骤，每次只处理一个章节，内容需要是最新的Kotlin知识，如涉及图表，请采用mermaid格式。


请你来撰写一篇10万字,面向已经精通TypeScript/React/Vue/前端开发，了解Python/浏览器/Playwright/MySQL数据库的中级全栈开发人员的 网络爬虫高级 教程。请你先列出教程大纲，将大纲写在放在 `E:\DehazeSystem\dehaze-doc\docs\前端开发\爬虫` 根目录下，作为当前任务的记忆，每次写完一个章节都可以回顾一下该文档。

你可以在适当的时机采用 sequential thinking mcp 进行规划、context7 / fetch mcp 获取最新的知识。

然后是具体的教程文档要求，采用markdown格式，分章节写入到目录 `E:\DehazeSystem\dehaze-doc\docs\前端开发\爬虫` 下，在该文件夹下每个部分创建一个文件夹，在每个部分内部，每个章节为一个md文件，文件名采用章节名。内容你需要分多个步骤，每次只处理一个章节，内容需要是最新的爬虫知识，如涉及图表，请采用mermaid格式。
具体的教程内容可以参考但不限于这些内容：网页抓取的原理和技术、爬虫与反爬原理、常见反爬措施（动态网页加载封禁IP、验证码、JS加密/混淆，复杂的加密算法，APP防破解/加固等）的原理以及解决方法，大规模数据抓取以及清洗，分布式/多线程网络爬虫以及实际开发中可能遇到的优化调度、并发等问题，大型网站实战爬取
你应该先给我列出大纲，等待我的修改和审核，随后再开始具体教程的编写。

以下是剩余的所有部分大纲，继续第五部分：实战项目开发

🎨 第五部分：实战项目开发（约1.5万字）
第21章：项目架构设计（3000字）
21.1 MVVM架构模式
21.2 模块化设计原则
21.3 代码规范与约定
21.4 版本控制策略
21.5 CI/CD集成方案

第22章：电商应用开发（4000字）
22.1 需求分析与功能设计
22.2 首页与商品展示
22.3 购物车与订单系统
22.4 支付集成实现
22.5 用户中心与设置

第23章：社交应用开发（4000字）
23.1 即时通讯功能
23.2 好友系统设计
23.3 动态发布与浏览
23.4 消息推送实现
23.5 多媒体分享功能

第24章：工具应用开发（4000字）
24.1 记事本应用
24.2 天气预报应用
24.3 文件管理器
24.4 系统监控工具
24.5 个性化设置
