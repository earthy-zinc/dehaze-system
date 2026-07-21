# dehaze_flutter

图像去雾系统的 Flutter 客户端，提供 iOS/Android/Web/Desktop 全平台支持。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Flutter前端架构文档.md)。

## 技术栈

- Flutter 3.35+ / Dart 3.9+
- Riverpod（状态管理）
- GoRouter（路由管理，含路由守卫）
- Dio（网络请求，拦截器链：Auth → Response → Retry → Error）
- SharedPreferences（Token 持久化）
- json_serializable + build_runner（序列化）

## 快速开始

```bash
# 安装依赖
flutter pub get

# 生成序列化代码
dart run build_runner build --delete-conflicting-outputs

# 运行（需后端 Java 服务运行在 127.0.0.1:8989）
flutter run -d chrome --web-port 5177    # Web（固定端口 5177）
flutter run -d windows                    # Windows
flutter run -d android                    # Android
```
