# React Native (dehaze-react-native)

基于 React Native + TypeScript 构建的移动端图像去雾应用，支持 iOS 和 Android 双平台。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 功能特性

- 🔐 **JWT 认证**：登录/Token 管理/权限校验
- 🏠 **首页展示**：Hero 区、算法介绍、功能特性、工作流、技术规格、CTA
- 🖼️ **图像输入**：本地上传、相机拍照、样张画廊、快速开始、历史记录
- 🎯 **算法选择**：算法卡片、算法树、对比栏、对比弹窗
- ⚙️ **去雾处理**：参数面板、处理进度、结果预览
- 📊 **效果对比**：并排对比、重叠对比、放大镜、滤镜、指标评估
- 📁 **数据集管理**：列表、详情、图片网格、类型筛选、图片查看器
- 📋 **任务历史**：历史任务列表与详情

## 2. 项目结构

```
dehaze-react-native/
├── android/                       # Android 原生工程
├── ios/                           # iOS 原生工程
├── src/
│   ├── App.tsx                    # 应用入口
│   ├── api/                       # API 接口封装
│   ├── components/                # 通用组件（Badge/Button/Card/EmptyState/Icon/ImageLoader/Modal/Section）
│   ├── config/                    # env、sdk 配置
│   ├── enums/                     # 枚举（CacheEnum）
│   ├── hooks/                     # 通用 hooks（useAnimation/useResponsive）
│   ├── layout/                    # 主布局（MainLayout/AppHeader/BottomTabBar/DrawerMenu/SideNav/MenuConfig）
│   ├── pages/                     # 业务页面
│   │   ├── home/                  # 首页（Hero/Algorithm/Features/FinalCTA/Showcase/TechSpecs）
│   │   ├── login/                 # 登录
│   │   ├── image-input/           # 图像输入（CameraCapture/UploadArea/SampleGallery/HistoryList）
│   │   ├── algorithm-select/      # 算法选择（AlgorithmCard/AlgorithmTree/CompareBar/CompareModal）
│   │   ├── processing/            # 去雾处理（ParamsPanel/ProcessingProgress/ResultPreview）
│   │   ├── compare/               # 效果对比（Filter/Magnifier/Metrics/Overlay/SideBySide）
│   │   ├── dataset/               # 数据集管理（DatasetListSection/ImageGrid/ImageViewer/TypeFilter）
│   │   ├── algorithm/             # 算法列表
│   │   └── task/                  # 任务历史
│   ├── routes/                    # 路由配置（config/navigator/types/utils）
│   ├── store/                     # 全局状态（AuthContext/AlgorithmContext/ImageContext）
│   ├── theme/                     # 主题（colors/spacing/typography）
│   ├── types/                     # 类型定义（algorithm/evaluation/image/processing）
│   └── utils/                     # storage/tokenStore
├── index.js                       # 应用入口文件
└── package.json
```

## 3. 架构设计

- **框架**：React Native + TypeScript
- **状态管理**：Context API（AuthContext/AlgorithmContext/ImageContext）
- **网络层**：`config/sdk.ts` + `api/` 目录封装，统一 Token 注入
- **路由**：自研路由配置（`routes/config.tsx`），支持导航器与类型安全
- **主题**：统一的 colors/spacing/typography 设计令牌
- **布局**：移动端 BottomTabBar + 桌面端 SideNav/DrawerMenu 响应式适配

## 4. 系统亮点

- 完整的移动端去雾业务链路（输入→选算法→处理→对比→历史）
- 自研路由系统支持类型安全导航
- 组件化程度高，通用组件覆盖 Badge/Button/Card/Modal/Icon 等基础能力
- 通过 `useResponsive` hook 实现移动端/平板/桌面端响应式适配
