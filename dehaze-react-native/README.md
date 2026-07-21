# dehaze-react-native

基于 React Native 的图像去雾系统移动应用。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/ReactNative前端架构文档.md)。

## 技术栈

- React Native 0.81.4
- React 19.1.0
- TypeScript
- react-native-safe-area-context

## 环境要求

- Node.js >= 20
- pnpm >= 8
- Android Studio 或 Xcode（用于运行原生应用）

## 快速开始

```bash
# 安装依赖
yarn

# 运行 Android
yarn android
# 或先启动 Metro，再运行原生构建:
yarn start
yarn react-native run-android

# 运行 iOS
yarn ios
# 或先启动 Metro，再运行原生构建:
yarn start
yarn react-native run-ios

# 测试与检查
yarn test
yarn lint
```
