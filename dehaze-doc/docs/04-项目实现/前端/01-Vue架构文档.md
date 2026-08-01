# Vue3 前端 (dehaze-front-vue)

基于深度学习的在线实时响应的图像去雾系统 Web 前端，主要功能是改善受到雾霾影响的图像质量，从而实现图像去雾的目标。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 架构图

```mermaid
flowchart TB
    subgraph View["视图层"]
        Pages["页面组件 (views/)"]
        Layout["布局组件"]
        CoreComp["核心业务组件 (dehaze/compare/algorithm)"]
        BaseComp["基础模块组件 (FavoriteButton/RecommendationWidget/MyFavoritesPage)"]
        CommonComp["通用组件 (Waterfall/Magnifier/DraggableLine/ImportExportToolbar)"]
    end

    subgraph State["状态管理层"]
        Pinia["Pinia Store"]
        Modules["store/modules (user/permission/settings/favorite)"]
        Persist["localStorage 持久化"]
    end

    subgraph API["API 层"]
        Request["axios 封装 (拦截器)"]
        ApiModules["api/ 模块"]
    end

    subgraph Router["路由层"]
        StaticRoutes["静态路由"]
        DynamicRoutes["动态路由 (按角色加载)"]
        Guards["导航守卫"]
    end

    subgraph Backend["后端"]
        REST["RESTful API"]
        WS["WebSocket (去雾进度)"]
    end

    Pages --> Pinia
    Layout --> Pinia
    CoreComp --> Pinia
    BaseComp --> Pinia
    CommonComp --> Pinia
    Pinia --> ApiModules
    ApiModules --> Request
    Request --> REST
    Router --> Pages
    WS --> Pinia
```

## 2. 目录结构

- 分层清晰：
  - `views` - 存放页面组件
  - `components` - 封装复用组件
  - `api` - 统一接口管理
- 模块化设计：
  - `store/modules` - 划分用户、权限、设置等独立模块
  - `enums` - 集中管理枚举类型
  - `typings` - 定义类型声明文件
- 功能模块化：
  - `compare` - 实现图像对比功能
  - `dataset` - 管理数据集
  - `evaluation` - 展示算法评估指标
- 组件复用：
  - `Waterfall` - 实现瀑布流布局
  - `Magnifier` - 封装画布缩放功能
  - `DraggableLine` - 实现对比图层拖拽
  - `ImportExportToolbar` - 通用导入导出工具栏（含导入弹窗、导出弹窗、任务列表抽屉）
  - `FavoriteButton` - 跨模块可复用收藏按钮（v2.0 新增）
  - `RecommendationPanel` - 图像分析推荐面板（v2.0 新增）
- 组合式 API：
  - `usePagination` - 通用分页逻辑
  - `useTableSelection` - 表格多选逻辑
  - `useDebounce` / `useDebouncedRef` - 防抖函数与响应式防抖
  - `useAsyncTask` - 异步任务 loading/error/data 三态管理
  - `useConfirm` / `useDeleteConfirm` - 二次确认弹窗

## 3. 核心模块

| 模块 | 功能描述 | 技术要点 |
|--------|-----------------------------------------------------|---------|
| 用户系统 | 支持角色/权限管理、多级部门树、Session 认证与自动续期 | Pinia + localStorage Token 持久化 |
| 数据集管理 | 瀑布流展示 + 懒加载、图片 MD5 校验、图片数量统计 | 缩略图 + 瀑布流 + 懒加载 |
| 去雾处理 | 单张/批量去雾、参数调节（通用+算法专属）、参数预设管理、处理历史 | WebSocket 实时进度推送 + 异步任务状态轮询 |
| 算法选择 | 树形算法结构、关键词/拼音搜索、多算法对比（最多3个）、自定义图片测试 | 拼音预计算字段 + 雷达图/柱状图可视化对比 |
| 效果对比 | 6种对比模式（并排/重叠/放大镜/滤镜/指标/算法信息）、对比报告导出 | CSS clip-path 重叠对比 + ECharts 指标可视化 |
| 收藏管理 | 跨模块统一收藏（算法/处理结果/数据集/图片/预设）、"我的收藏"聚合页 | FavoriteButton 可复用组件 + FavoriteAPI (SDK) |
| 推荐管理 | 图像特征分析推荐（7维特征）、管理员规则配置、推荐效果报表 | RecommendationPanel 组件 + RecommendationAPI (SDK) |
| 通用导入导出 | 为各列表页面提供统一的 Excel/CSV 导入导出能力，复用任务列表查看进度 | ImportExportToolbar 组件 |
| 系统配置 | 主题色切换、暗黑模式、布局模式（侧边/顶部/混合）、水印开关 | CSS 变量 + 组件化管理 |

## 4. 组件分层

```mermaid
flowchart TB
    subgraph PageLayer["页面组件层"]
        DehazePage["去雾处理页"]
        ComparePage["效果对比页"]
        AlgoSelectPage["算法选择页"]
        FavoritesPage["我的收藏页"]
    end

    subgraph CoreBiz["核心业务组件层"]
        DehazeComp["去雾组件 (进度/参数/历史)"]
        CompareComp["对比组件 (重叠/放大镜/滤镜/指标)"]
        AlgoComp["算法组件 (树形/搜索/对比)"]
    end

    subgraph BaseBiz["基础模块组件层"]
        FavoriteBtn["FavoriteButton 收藏按钮"]
        RecommendPanel["RecommendationPanel 推荐组件"]
    end

    subgraph Common["通用组件层"]
        Waterfall["Waterfall 瀑布流"]
        Magnifier["Magnifier 放大镜"]
        DraggableLine["DraggableLine 拖拽分隔线"]
        ImportExport["ImportExportToolbar 导入导出"]
    end

    PageLayer --> CoreBiz
    PageLayer --> BaseBiz
    PageLayer --> Common
    CoreBiz --> BaseBiz
    CoreBiz --> Common
```

## 5. 关键技术决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 状态管理 | Pinia | Vue3 官方推荐，模块化 Store，支持 DevTools |
| 认证方案 | Session + Token 持久化 | localStorage 持久化 Token，请求拦截器自动注入 |
| 路由方案 | 静态路由 + 动态路由 | 登录后动态获取用户菜单，按角色展示不同菜单 |
| 响应式布局 | CSS 变量 + 弹性盒子 | 同一套代码适配桌面端和移动端，支持多种布局模式切换 |
| 图片上传 | 前端 MD5 校验 + 并发上传 | 前端预校验减轻后端压力，并发请求提升速度 |
| 图像对比 | CSS clip-path + Canvas | 重叠对比 + 放大镜细节查看 |
| 组件分层 | 基础组件 / 业务组件 / 布局组件 | 清晰分层，提高可维护性 |
| 跨模块收藏状态同步 | Pinia 全局 favorite Store | 收藏状态在算法列表、处理历史、对比页面等多处复用，全局 Store 保证状态一致性，避免组件间 prop 透传 |
| 6种对比模式切换 | 动态组件 + `<component :is>` | 对比模式通过组件名动态切换，避免 v-if 嵌套，模式间独立维护状态 |
| 推荐组件可嵌入设计 | RecommendationWidget 独立组件 | 可嵌入算法选择页、去雾处理页、效果对比页，通过 props 接收场景参数控制推荐策略 |

## 6. 模块间交互

- 通过 RESTful API 调用 Java/Go/Python 后端
- 通过 WebSocket 长连接获取去雾进度实时推送
- 与 React 前端共享同一套后端 API
