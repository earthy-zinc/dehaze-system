# Dehaze Flutter App

图像去雾系统的 Flutter 客户端，提供 iOS/Android/Web/Desktop 全平台支持。

## 功能特性

- 🔐 **JWT 认证**：完整的登录/登出/验证码/Token 刷新流程
- 🌫️ **去雾处理**：图像输入 → 算法选择 → 去雾处理 → 效果对比 完整链路
- 📊 **效果对比**：并排对比、重叠对比、放大镜、滤镜调节、指标评估、算法信息
- 📁 **数据集管理**：数据集列表、详情、图片浏览、类型筛选
- 👤 **用户中心**：用户信息、角色权限、处理历史
- 📱 **响应式布局**：移动端底部导航 + 桌面端侧边栏

## 技术栈

- **Flutter**: 3.35+ / **Dart**: 3.9+
- **状态管理**: Riverpod
- **路由管理**: GoRouter（含路由守卫）
- **网络请求**: Dio（拦截器链：Auth → Response → Retry → Error）
- **本地存储**: SharedPreferences（Token 持久化）
- **序列化**: json_serializable + build_runner

## 项目结构

```
lib/
├── main.dart                          # 入口
├── core/                              # 核心基础设施
│   ├── auth/                          # 认证错误处理
│   ├── constants/                     # 常量（API路径、存储Key）
│   ├── network/                       # 网络层（Dio + 拦截器 + 响应模型）
│   └── storage/                       # Token 存储
├── models/                            # 共享数据模型
├── services/                          # API 服务层
├── providers/                         # Riverpod Providers
├── router/                            # 路由配置
├── layout/                            # 主布局 + 菜单
├── theme/                             # 主题
└── pages/                             # 功能页面
    ├── home/                          # 首页
    ├── login/                         # 登录页
    ├── image_input/                   # 图像输入
    ├── algorithm_select/              # 算法选择
    ├── processing/                    # 去雾处理
    ├── comparison/                    # 效果对比（6个子页面）
    ├── dataset/                       # 数据集管理
    ├── profile/                       # 用户中心
    └── task_history/                  # 处理历史
```

## 快速开始

```bash
# 安装依赖
flutter pub get

# 生成序列化代码
dart run build_runner build --delete-conflicting-outputs

# 运行（需后端 Java 服务运行在 127.0.0.1:8989）
flutter run -d chrome    # Web
flutter run -d windows   # Windows
flutter run -d android   # Android
```

## 后端配置

- API 基础路径：`http://127.0.0.1:8989/api/v1`
- 默认账号：`admin` / `123456`
- Windows 环境必须使用 `127.0.0.1`（非 `localhost`）

## 许可证

MIT
