# dehaze-front-vue 组件库

## 组件分层

### 通用组件层（Common）
基础 UI 组件，无业务耦合

| 组件 | 路径 | 说明 | Props 数 | Events | Slots |
|------|------|------|----------|--------|-------|
| Waterfall | Waterfall/ | 瀑布流布局，支持懒加载、无限滚动、响应式列数 | 15 | afterRender | - |
| LazyImg | LazyImg/ | 图片懒加载组件 | 3 | - | - |
| Loading | Loading/ | 加载动画/遮罩 | 2 | - | - |
| SvgIcon | SvgIcon/ | SVG 图标组件，基于 inline-svg | 4 | - | - |
| Pagination | Pagination/ | 分页组件，封装 el-pagination | 8 | pagination | - |
| Hamburger | Hamburger/ | 汉堡菜单切换按钮 | 3 | change | - |
| LangSelect | LangSelect/ | 语言选择下拉 | 0 | - | - |
| SizeSelect | SizeSelect/ | 视图尺寸选择下拉 | 0 | - | - |
| IconSelect | IconSelect/ | 图标选择器弹窗 | 2 | select, update:visible | - |

### 核心业务组件层（Core Business）
与去雾业务紧密相关的组件

| 组件 | 路径 | 说明 | 依赖 SDK |
|------|------|------|----------|
| AlgorithmToolBar | AlgorithmToolBar/ | 算法参数配置工具栏 | AlgorithmAPI |
| EffectDisplay | EffectDisplay/ | 算法效果前后对比展示 | ImageStore |
| Magnifier | Magnifier/ | 图片放大镜查看效果 | ImageStore |
| RatingCard | RatingCard/ | 算法效果评价弹窗 | FeedbackAPI, FileAPI |

### 基础模块组件层（Base Business）
跨基础模块复用的业务组件

| 组件 | 路径 | 说明 | 依赖 SDK |
|------|------|------|----------|
| ImportExportToolbar | ImportExportToolbar/ | 通用导入导出工具栏 | ImportAPI, ExportAPI |
| DataList | DataList/ | 数据列表展示 | - |
| Dictionary | Dictionary/ | 字典管理面板 | DictAPI |
| DraggableLine | DraggableLine/ | 可拖拽排序的行 | - |
| InfiniteFlowList | InfiniteFlowList/ | 无限滚动列表 | - |
| Camera | Camera/ | 摄像头捕获组件 | CameraAPI |
| DatasetImageSelect | DatasetImageSelect/ | 数据集图片选择器 | DatasetAPI |
| ExampleImageSelect | ExampleImageSelect/ | 示例图片选择器 | - |

### 图像相关组件层（Image）
专用图像展示/对比组件

| 组件 | 路径 | 说明 |
|------|------|------|
| SingleImageShow | SingleImageShow/ | 单张图片展示 |
| ParallelImageShow | ParallelImageShow/ | 多图并行对比展示，支持放大镜 |
| OverlapImageShow | OverlapImageShow/ | 双图重叠对比，支持滑块和放大镜 |
| ParallelImageUpload | ParallelImageUpload/ | 多图上传（雾图/预测图/原图） |
| Upload (SingleUpload) | Upload/ | 单图上传，支持 MD5 秒传 |
| Upload (MultiUpload) | Upload/ | 多图上传，支持 MD5 秒传 |

### 布局/系统组件层（Layout/System）

| 组件 | 路径 | 说明 |
|------|------|------|
| TitleBar | TitleBar/ | Electron 窗口标题栏（最小化/最大化/关闭） |
| Breadcrumb | Breadcrumb/ | 面包屑导航 |
| AppLink | AppLink/ | 应用内链接跳转 |
| LongitudinalWaterfall | LongitudinalWaterfall/ | 纵向瀑布流布局 |

## 使用规范

### 命名约定
- 组件名使用 PascalCase
- 目录名使用 PascalCase
- 入口文件统一为 index.vue（Upload 除外，包含 SingleUpload.vue / MultiUpload.vue）

### 组件文档规范
每个组件应包含：
- Props 完整类型定义
- Events 说明
- Slots 说明
- 使用示例（最小可运行代码）

### 分层使用原则
1. **通用组件层**可在任何场景下直接使用，无业务依赖
2. **核心业务组件层**仅在与去雾算法相关页面中使用
3. **基础模块组件层**在对应基础模块（数据集、字典等）中使用
4. **图像相关组件层**仅在需要图像展示/上传的场景中使用
5. **布局/系统组件层**用于整体布局和框架级功能

### 样式规范
- 组件样式使用 scoped 隔离
- CSS 变量优先使用 `var(--el-*)` Element Plus 设计变量
- 自定义颜色通过 `settingStore.themeColor` 获取主题色
