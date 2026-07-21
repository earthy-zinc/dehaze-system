# Taro 多端 (dehaze-taro)

基于 Taro 4 + React + TypeScript 构建的多端图像去雾应用，一份代码可编译到微信小程序、H5、支付宝小程序等多个平台。

> 构建运行、测试命令、部署说明见项目根目录 [README](/README.md)。

## 1. 功能特性

- 🔐 **JWT 认证**：登录/注册/Token 刷新/权限校验
- 🏠 **首页展示**：算法介绍、工具矩阵、工作流演示、技术规格
- 🖼️ **图像输入**：本地上传、相机拍照、样张画廊、历史记录
- 🎯 **算法选择**：算法列表、参数配置、算法说明
- ⚙️ **去雾处理**：实时进度、结果预览、参数调节
- 📊 **效果对比**：并排对比、重叠对比、放大镜、滤镜、指标评估、算法信息
- 📁 **数据集管理**：列表、详情、图片瀑布流、类型筛选
- 🛠️ **系统管理**：用户、部门、角色、菜单、字典管理

## 2. 项目结构

```
dehaze-taro/
├── config/                        # Taro 编译配置
├── src/
│   ├── app.tsx                    # 应用入口
│   ├── app.config.ts              # 应用配置（页面路由、tabBar）
│   ├── components/                # 通用组件
│   │   ├── common/                # EmptyState/FilterTabs/ImageCard/ImageViewer/SearchBar
│   │   ├── compare/               # AlgorithmInfoCard/CompareToolbar
│   │   └── system/                # PermissionGuard
│   ├── config/                    # api、menu 配置
│   ├── hooks/                     # 业务 hooks（auth/permission/dept/role/menu/user/dict/layout/system）
│   ├── layout/                    # navbar/sidebar/tabbar 布局
│   ├── pages/                     # 业务页面
│   │   ├── home/                  # 首页（Hero/算法/工具/工作流/技术规格/CTA）
│   │   ├── login/                 # 登录注册
│   │   ├── image-input/           # 图像输入（上传/拍照/样张/历史）
│   │   ├── algorithm-select/      # 算法选择
│   │   ├── processing/            # 去雾处理
│   │   ├── side-by-side/          # 并排对比
│   │   ├── overlay/               # 重叠对比
│   │   ├── magnifier/             # 放大镜
│   │   ├── filter/                # 滤镜调节
│   │   ├── metrics/               # 指标评估
│   │   ├── algorithm/             # 算法列表
│   │   ├── dataset/               # 数据集管理
│   │   ├── task/                  # 任务历史
│   │   ├── dashboard/             # 仪表盘
│   │   └── system/                # 系统管理（user/dept/role/menu/dict）
│   ├── router/                    # 路由配置
│   ├── stores/                    # 全局状态
│   ├── utils/                     # permission/request/storage/upload
│   └── types/                     # 类型定义
└── package.json
```

## 3. 架构设计

- **框架**：Taro 4 + React 18 + TypeScript
- **状态管理**：Context + 自定义 hooks（useAuth/usePermission/useUserManagement 等）
- **网络层**：封装 `utils/request.ts`，统一 Token 注入与错误处理
- **样式**：Less + 全局变量，支持多端样式适配
- **权限**：`PermissionGuard` 组件 + `usePermission` hook 实现按钮级权限控制

## 4. 多端适配

- 移动端竖屏优化，适配手机和平板
- 微信小程序、H5、支付宝小程序等多端编译
- 通过 `app.config.ts` 配置各端差异
