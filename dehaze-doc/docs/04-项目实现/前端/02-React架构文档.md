# React 前端 (dehaze-front-react)

基于深度学习的在线实时响应的图像去雾系统 Web 前端，主要功能是改善受到雾霾影响的图像质量。采用 React + TypeScript + Vite + Ant Design + Redux Toolkit 构建，并通过 Electron 提供桌面端应用。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 功能特性

### 1.1 用户管理模块

- 支持用户注册/登录/权限管理
- 角色-权限-菜单三级权限控制（RoleAPI 与 AuthAPI）
- 用户信息加密传输（Token 认证机制）

### 1.2 数据集管理

- 数据集分页展示（`DatasetList` 组件）
- 数据集详情页支持图片瀑布流展示（`Waterfall` 组件）
- 支持数据集导入导出（`DatasetAPI.export` 接口）

### 1.3 图像处理功能

- 实时摄像头捕获（`Camera` 组件）
- 图像叠加对比（`OverlapImageShow` 组件）
- 放大镜效果（`Magnifier` 组件）
- 图像参数调节（对比度/亮度控制）

### 1.4 算法集成

- 算法工具栏支持参数配置（`AlgorithmToolBar` 组件）
- 模型选择与预测结果可视化（`ModelAPI` 接口）

## 2. 技术栈

| 类别 | 技术 |
|------|------|
| 框架 | React 18 + TypeScript |
| 构建 | Vite |
| UI | Ant Design |
| 状态管理 | Redux Toolkit（模块化 slice） |
| 样式 | UnoCSS 原子化方案 |
| 桌面端 | Electron（代码位于 `electron` 目录） |

## 3. 架构设计

- **状态管理**：Redux Toolkit 模块化划分，`store/modules` 下按业务领域拆分多个 slice
- **桌面端集成**：通过 Electron 集成桌面端能力，相关代码位于 `electron` 目录（`electron/main/index.ts`、`electron/preload/index.ts`）
- **样式方案**：UnoCSS 原子化 + Ant Design 实现统一视觉风格
- **组件化**：Camera、OverlapImageShow、Magnifier、AlgorithmToolBar、DatasetList、Waterfall 等独立可复用组件

## 4. 与 Vue3 前端的定位差异

- **Vue3 前端（dehaze-front-vue）**：系统主前端，功能最完整，覆盖全部业务模块
- **React 前端（dehaze-front-react）**：技术栈对照实现，验证 React 生态下的等价能力，含 Electron 桌面端

## 5. 后续规划

- 补全与 Vue3 前端对齐的业务模块
- 优化 Electron 桌面端打包与自动更新流程
- 探索微前端架构（qiankun）集成方案
