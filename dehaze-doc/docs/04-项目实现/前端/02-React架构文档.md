# React 前端 (dehaze-front-react)

基于深度学习的在线实时响应的图像去雾系统 Web 前端，采用 React + TypeScript + Vite + Ant Design + Redux Toolkit 构建，并通过 Electron 提供桌面端应用。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 架构图

```mermaid
flowchart TB
    subgraph View["视图层"]
        Pages["页面组件 (pages/)"]
        CoreComp["核心业务组件 (Camera/OverlapImageShow/Magnifier/AlgorithmToolBar)"]
        CommonComp["通用组件 (Waterfall/ImportExportToolbar/FavoriteButton/RecommendationWidget)"]
    end

    subgraph State["状态管理"]
        Redux["Redux Toolkit"]
        Slices["store/modules (按业务拆分 slice)"]
    end

    subgraph SDK["API 层 (dehaze-sdk-js)"]
        ApiModules["API 模块 (RoleAPI/AuthAPI/ModelAPI/ImportExportAPI/FavoriteAPI/RecommendationAPI)"]
    end

    subgraph Style["样式层"]
        UnoCSS["UnoCSS 原子化"]
        AntDesign["Ant Design 组件库"]
    end

    subgraph Electron["桌面端 (项目根 electron/)"]
        Main["Main Process"]
        Preload["Preload Script"]
    end

    subgraph Backend["后端"]
        REST["RESTful API"]
    end

    Pages --> Redux
    CoreComp --> Redux
    CommonComp --> Redux
    Redux --> ApiModules
    ApiModules --> REST
    Pages --> Style
    CoreComp --> Style
    CommonComp --> Style
    Electron --> Pages
```

## 2. 目录结构

- `pages/` - 页面组件（React 约定使用 pages 而非 views）
- `components/` - 可复用组件
- `layout/` - 布局组件
- `hooks/` - 自定义 Hook
- `router/` - 路由配置
- `store/modules/` - Redux Toolkit slice（app/dataset/favorite/imageShow/permission/settings/task/user）
- `enums/` - 枚举类型
- `typings/` - 类型声明
- `utils/` - 工具函数
- `styles/` - 全局样式
- `assets/` - 静态资源
- 桌面端代码位于项目根目录 `electron/`（main/index.ts、preload/index.ts），不在 src 下
- API 统一由 `dehaze-sdk-js` 提供

## 3. 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | React 18 + TypeScript |
| 构建 | Vite |
| UI | Ant Design |
| 状态管理 | Redux Toolkit（模块化 slice） |
| 样式 | UnoCSS 原子化方案 |
| 桌面端 | Electron |

## 4. 核心模块

### 4.1 用户管理模块

- 支持用户注册/登录/权限管理
- 角色-权限-菜单三级权限控制
- Token 认证机制

### 4.2 数据集管理

- 数据集分页展示（数据集列表页）
- 数据集详情页支持图片瀑布流展示（Waterfall 组件）
- 数据集导出通过通用导入导出框架实现

### 4.3 通用导入导出

- ImportExportToolbar 组件为各列表页面提供统一的导入/导出/模板下载按钮
- 支持用户/角色/部门/菜单/字典/数据集/算法模块的 Excel/CSV 导入导出
- 复用任务列表查看异步导入导出任务进度

### 4.4 图像处理功能

- 实时摄像头捕获（Camera 组件）
- 图像叠加对比（OverlapImageShow 组件）
- 放大镜效果（Magnifier 组件）
- 图像参数调节（对比度/亮度控制）

### 4.5 算法集成

- 算法工具栏支持参数配置（AlgorithmToolBar 组件）
- 模型选择与预测结果可视化

### 4.6 去雾处理模块

- 单张/批量去雾处理，参数调节（通用参数 + 算法专属参数）
- 参数预设管理（管理员预设 + 用户自定义预设）
- 处理历史列表/网格视图切换，支持筛选和重新处理
- 轮询 TaskAPI 获取异步处理进度

### 4.7 效果对比模块

- 多种对比模式：并排、重叠、放大镜、滤镜、指标、算法信息
- ECharts 指标可视化（雷达图 + 柱状图）
- 对比报告异步导出（复用任务管理模块）

### 4.8 算法选择模块

- 树形算法结构展示，关键词/拼音多维度搜索
- 最多 3 个算法多维度对比（表格 + 柱状图 + 雷达图）
- 自定义图片测试，临时预览不入历史记录

### 4.9 收藏管理模块

- FavoriteButton 可复用组件，通过 `targetType` prop 适配算法/处理结果/数据集
- 我的收藏页（`/favorite/my`），按类型筛选 + 排序 + 搜索
- Redux favoriteSlice 管理全局收藏状态，useFavorite Hook 封装收藏操作

### 4.10 推荐管理模块

- RecommendationWidget 可嵌入组件，通过 `imageId`/`imageUrl` prop 接入图像
- 调用 `RecommendationAPI.analyze()` 进行 7 维图像特征分析（雾霾浓度/场景类型/光照/复杂度/分辨率/噪声）
- Top 3 推荐算法展示（匹配度进度条 + 推荐理由 + 评分），支持一键选用
- 推荐效果反馈（有用/无用）通过 `RecommendationAPI.submitFeedback()` 上报
- 管理员推荐规则管理页（`/recommendation/rules`），支持权重/启用/算法多选配置

### 4.11 算法管理详情页

- 算法详情页（`/algorithm/detail`），Tab 三级信息架构（基本/技术/运营信息）
- 调用 `AlgorithmAPI.getAlgorithmInfoById()` + `AlgorithmAPI.getMonitorData()` 获取详情和监控数据

### 4.12 算法选择推荐集成

- 算法选择页支持图像分析推荐（Tab 切换：快速推荐 / 图像分析推荐）
- 输入图片 URL → `RecommendationAPI.analyze()` → `getAlgorithmRecommendations()` → Top 3 匹配算法
- 推荐算法可一键选用，与手动选择流程一致

## 5. 组件分层

```mermaid
flowchart TB
    subgraph PageLayer["页面组件层"]
        DehazePage["去雾处理页"]
        ComparePage["效果对比页"]
        AlgoSelectPage["算法选择页"]
        FavoritesPage["我的收藏页"]
    end

    subgraph CoreBiz["核心业务组件层"]
        Camera["Camera 摄像头"]
        Overlap["OverlapImageShow 叠加对比"]
        Magnifier["Magnifier 放大镜"]
        AlgoToolBar["AlgorithmToolBar 算法工具栏"]
    end

    subgraph Common["通用组件层"]
        FavoriteBtn["FavoriteButton 收藏按钮"]
        RecommendWidget["RecommendationWidget 推荐组件"]
        Waterfall["Waterfall 瀑布流"]
        ImportExport["ImportExportToolbar 导入导出"]
    end

    PageLayer --> CoreBiz
    PageLayer --> Common
    CoreBiz --> Common
```

## 6. 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 框架 | React 18 | 与 Vue3 前端形成对照，验证 React 生态等价能力 |
| 状态管理 | Redux Toolkit | 模块化 slice，与 React 生态深度集成 |
| 样式方案 | UnoCSS | 原子化 CSS，按需生成，无运行时开销 |
| 桌面端 | Electron | 提供跨平台桌面应用能力 |
| 构建工具 | Vite | 快速开发服务器，HMR 热更新 |
| 收藏状态同步 | Redux favoriteSlice + useFavorite Hook | 组件通过 useSelector 订阅收藏状态，保证跨页面收藏状态一致 |
| 对比模式 | 条件渲染 + 独立组件 | 每个对比模式组件独立管理状态，按需组合 |
| 推荐组件可嵌入设计 | RecommendationWidget + props 驱动 | 通过 scene prop 控制推荐策略，可嵌入算法选择/去雾处理/效果对比页面 |
| 处理进度获取 | 轮询 TaskAPI | 异步去雾任务通过定时轮询获取状态 |

## 7. 设计原则与约束

- **单一数据源**：所有 API 调用统一通过 `dehaze-sdk-js`，项目内不重复封装接口层
- **组件分层约束**：页面组件不直接调用 API，通过 Redux 或 Hook 间接访问；通用组件不耦合具体业务逻辑
- **Hooks 模式复用**：跨组件复用的逻辑抽取为自定义 Hook，与 Vue composables 对齐
- **与 Vue 端功能对齐**：作为同一系统的并行实现，核心功能与 Vue 前端保持一致

## 8. 模块间交互

- 通过 RESTful API 调用 Java/Go/Python 后端
- 异步去雾任务通过轮询 TaskAPI 获取处理进度
- 与 Vue3 前端共享同一套后端 API，功能完整度对照实现
