# Dehaze Flutter App

一个基于 Flutter 开发的图像去雾应用，提供多平台支持。

## 功能特性

- 🌫️ 图像去雾处理
- 📱 多平台支持（Web、Android、iOS、Windows、Linux、macOS）
- 🎨 现代化 UI 设计
- 🔄 实时处理状态显示
- 📚 处理历史记录

## 技术栈

- **Flutter**: 3.35.7+
- **Dart**: 3.9.2+
- **状态管理**: Riverpod
- **路由管理**: GoRouter
- **网络请求**: Dio
- **本地存储**: SharedPreferences
- **网络检测**: Connectivity Plus

## 环境要求

- Flutter SDK: >=3.35.0
- Dart SDK: >=3.9.2
- Android Studio / VS Code
- Chrome (用于 Web 开发)

## 快速开始

### 1. 环境检查

```bash
flutter doctor
```

### 2. 安装依赖

```bash
flutter pub get
```

### 3. 运行应用

#### Web 平台（推荐开发使用）
```bash
flutter run -d chrome
```

#### Android 平台
```bash
flutter run -d android
```

#### Windows 桌面
```bash
flutter run -d windows
```

#### 其他平台
```bash
# 查看所有可用设备
flutter devices

# 选择特定设备运行
flutter run -d <device-id>
```

## 项目结构

```
lib/
├── app/                    # 应用核心
│   ├── app.dart           # 应用入口
│   ├── router/            # 路由配置
│   └── theme/             # 主题配置
├── core/                  # 核心功能
│   ├── errors/            # 错误处理
│   ├── network/           # 网络配置
│   └── utils/             # 工具类
├── features/              # 功能模块
│   └── dehaze/            # 去雾功能
│       ├── data/          # 数据层
│       ├── domain/        # 领域层
│       └── presentation/  # 表现层
├── services/              # 服务层
│   └── providers.dart     # Riverpod 提供者
└── main.dart              # 主入口文件
```

## 开发命令

```bash
# 热重载（运行时按 r）
# 热重启（运行时按 R）

# 分析代码
flutter analyze

# 运行测试
flutter test

# 构建应用
# Web
flutter build web

# Android
flutter build apk

# Windows
flutter build windows
```

## 配置说明

### 网络配置
网络请求配置在 `lib/core/network/api_config.dart` 文件中。

### 主题配置
应用主题在 `lib/app/theme/app_theme.dart` 文件中配置。

### 路由配置
路由配置在 `lib/app/router/config.dart` 文件中定义。

## 故障排除

### 常见问题

1. **依赖冲突**
   ```bash
   flutter clean
   flutter pub get
   ```

2. **运行时错误**
   检查网络连接和相关服务是否正常运行。

3. **设备连接问题**
   ```bash
   flutter devices
   ```

### 开发工具

- **Flutter DevTools**: http://127.0.0.1:9101
- **Dart VM Service**: 运行时在控制台查看地址

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。
