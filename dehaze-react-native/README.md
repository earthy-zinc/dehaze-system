# 去雾系统 - React Native 版本

基于 React Native 开发的图像去雾系统移动应用，用于上传图像并进行去雾处理。

## 项目介绍

这是一个使用 React Native 构建的移动端应用程序，专门用于图像去雾处理。该应用允许用户上传图片，并通过深度学习算法对雾霾图像进行处理，以获得更清晰的图像效果。

## 技术栈

- React Native 0.81.4
- React 19.1.0
- TypeScript
- react-native-safe-area-context

## 环境要求

- Node.js >= 20
- pnpm >= 8
- Android Studio 或 Xcode (用于运行原生应用)

## 安装步骤

1. 克隆项目代码:
   ```
   git clone <repository-url>
   ```

2. 进入项目目录:
   ```
   cd dehaze-react-native
   ```

3. 安装依赖:
   ```
   pnpm install
   ```

## 运行项目

### 运行在 Android 上

```
pnpm android
```

或者

```
pnpm start
```
然后在另一个终端执行:
```
pnpm react-native run-android
```

### 运行在 iOS 上

```
pnpm ios
```

或者

```
pnpm start
```
然后在另一个终端执行:
```
pnpm react-native run-ios
```

## 项目结构

```
dehaze-react-native/
├── android/                 # Android 原生代码
├── ios/                     # iOS 原生代码
├── src/                     # 应用源代码
│   ├── App.tsx             # 主应用组件
│   └── pages/              # 页面组件
├── index.ts                 # 应用入口文件
├── package.json             # 项目依赖和脚本配置
└── ...
```

## 可用脚本

- `pnpm android`: 构建并运行 Android 应用
- `pnpm ios`: 构建并运行 iOS 应用
- `pnpm start`: 启动 Metro 服务器
- `pnpm test`: 运行测试
- `pnpm lint`: 检查代码规范

## 开发说明

当前项目为基础模板，主要功能仍在开发中。后续将集成图像去雾的核心功能。

## 许可证

[待定]