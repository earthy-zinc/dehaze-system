# 图像去雾系统 Android 应用

将 dehaze-front-vue 完整重写为原生 Android 应用，使用 Java 语言开发。

## 技术栈

- Java + MVVM + Android Jetpack Components
- Retrofit + OkHttp（网络）、Glide（图片加载）
- Room（本地缓存）、DataStore（首选项）

## 功能模块

登录认证、数据集管理、算法浏览、图像去雾处理、效果对比（并排/叠加）、指标评估、系统管理

## 构建与运行

- 环境：Android Studio Flamingo+、JDK 8+、API 23+
- 后端地址配置：[DehazeApplication.java](app/src/main/java/com/pei/dehaze/DehazeApplication.java) 中 `setBaseUrl("http://10.0.2.2:8989")`（模拟器用 `10.0.2.2` 访问本机）

```bash
./gradlew build          # 构建
./gradlew installDebug   # 安装到设备
./gradlew testDebugUnitTest  # 运行测试
```

## SDK

项目包含 [dehaze-sdk-android](sdk/) 目录，基于 Retrofit2 + OkHttp3 封装的 API 客户端，支持 Token 自动管理、文件上传下载、异步回调。
# 图像去雾系统 Android 应用

[![License](https://img.shields.io/github/license/earthy-zinc/reading-note)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Android-green.svg)](https://www.android.com)
[![API](https://img.shields.io/badge/API-23%2B-brightgreen.svg?style=flat)](https://developer.android.com/studio/releases/platforms)
[![Status](https://img.shields.io/badge/status-completed-success)](#)

这是一个将 [dehaze-front-vue](../dehaze-front-vue)（桌面 Web）完整重写为原生 Android 应用的项目，使用 Java 语言开发。

## 功能特性

1. **用户认证** - 登录、注册、忘记密码
2. **数据集管理** - 查看和管理图像数据集
3. **算法管理** - 浏览和搜索去雾算法
4. **图像对比** - 并排和叠加方式对比处理前后图像
5. **图像评估** - 上传图像并进行去雾处理，查看评估指标
6. **图像展示** - 实时演示不同算法的去雾效果
7. **系统管理** - 管理用户、部门和角色信息

## 技术栈

- **语言**: Java
- **架构**: MVVM (Model-View-ViewModel)
- **框架**: Android Jetpack Components
- **网络库**: Retrofit + OkHttp
- **图片加载**: Glide
- **数据库**: Room (本地缓存)
- **数据存储**: DataStore (首选项存储)
- **测试框架**: JUnit, Mockito, Robolectric

## 项目状态

✅ **已完成** - 项目已按要求完成所有功能模块的开发和测试，达到了预期目标

## 架构设计

应用采用标准的 MVVM 架构模式，分为以下层级：

```
┌─────────────┐
│    View     │ ← UI层 (Activity, Fragment, View)
└─────────────┘
       ↓
┌─────────────┐
│  ViewModel  │ ← 数据处理和UI逻辑层
└─────────────┘
       ↓
┌─────────────┐
│ Repository  │ ← 数据仓库层
└─────────────┘
       ↓
┌─────────────┐
│ Data Source │ ← 数据源 (网络API, 本地数据库)
└─────────────┘
```

## 模块结构

```
com.pei.dehaze
├── ui                  # UI层
│   ├── login           # 登录模块
│   ├── dashboard       # 仪表盘模块
│   ├── dataset         # 数据集模块
│   ├── algorithm       # 算法模块
│   ├── compare         # 图像对比模块
│   ├── evaluation      # 图像评估模块
│   ├── presentation    # 图像展示模块
│   └── system          # 系统管理模块
├── repository          # 数据仓库层
├── model               # 数据模型
├── network             # 网络层
├── utils               # 工具类
├── common              # 公共组件
└── sdk                 # SDK封装
```

## 主要组件

- Navigation Component - 应用内页面导航
- ViewModel + LiveData - UI 状态管理
- RecyclerView - 列表展示
- ViewPager2 - 页面滑动切换
- DataBinding - 数据绑定
- CameraX - 拍照功能

## 构建和运行

### 环境要求

- Android Studio Flamingo 或更高版本
- JDK 8 或更高版本
- Android SDK API Level 23+
- Gradle 8.0+

### 构建步骤

1. 克隆项目到本地
```bash
git clone <repository-url>
```

2. 在 Android Studio 中打开项目

3. 同步 Gradle 依赖
```bash
./gradlew build
```

4. 运行应用
```bash
./gradlew installDebug
```

### 测试

运行单元测试:
```bash
./gradlew testDebugUnitTest
```

生成测试覆盖率报告:
```bash
./gradlew jacocoTestReport
```

报告位置: `app/build/reports/jacoco/jacocoTestReport/html/index.html`

## API 配置

应用默认连接本地开发服务器，地址配置在 [DehazeApplication.java](app/src/main/java/com/pei/dehaze/DehazeApplication.java) 中:

```java
DehazeSDK.Builder()
    .setBaseUrl("http://10.0.2.2:8989") // Android模拟器访问本机需要使用10.0.2.2
    .setDebug(true)
```

如果需要更改服务器地址，请修改此处配置。

## 权限说明

应用需要以下权限:

- INTERNET - 访问网络接口
- CAMERA - 拍照功能
- READ_EXTERNAL_STORAGE/WRITE_EXTERNAL_STORAGE - 读取和保存图像文件

## 兼容性

- 最低支持 Android 6.0 (API Level 23)
- 支持 Android 14 (API Level 34)
- 屏幕适配: 支持各种屏幕尺寸和分辨率

## SDK 说明

项目包含一个基于 Retrofit2、OkHttp3、Lombok 和 Timber 的 Android SDK，位于 [dehaze-sdk-android](dehaze-sdk-android) 目录中。

### SDK 功能特点

- 基于 Retrofit2 和 OkHttp3 实现网络请求
- 使用 Lombok 简化 Java 代码
- 集成 Timber 日志框架
- 提供完整的 API 接口封装
- 支持异步回调处理
- 支持文件上传和下载
- 自动处理 Token 认证
- 模块化 API 设计

### SDK 使用方式

#### 方式一：作为模块导入（推荐）

1. 将 `dehaze-sdk-android` 目录复制到你的 Android 项目根目录下
2. 在项目根目录的 `settings.gradle` 文件中添加：

```gradle
include ':dehaze-sdk-android'
```

3. 在需要使用 SDK 的模块（如 app 模块）的 `build.gradle` 文件中添加依赖：

```gradle
dependencies {
    implementation project(':dehaze-sdk-android')
}
```

#### 方式二：生成 AAR 文件并导入

1. 在 `dehaze-sdk-android` 目录下执行以下命令生成 AAR 文件：

```bash
./gradlew assembleRelease
```

2. 在你的 Android 项目中创建 `libs` 目录（如果不存在），将生成的 AAR 文件复制到该目录
3. 在需要使用 SDK 的模块（如 app 模块）的 `build.gradle` 文件中添加依赖：

```gradle
dependencies {
    implementation files('libs/dehaze-sdk-android-release.aar')
    // 注意：还需要手动添加 SDK 的依赖项
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.9.1'
    compileOnly 'org.projectlombok:lombok:1.18.22'
    annotationProcessor 'org.projectlombok:lombok:1.18.22'
}
```

### SDK 初始化

在 Application 的 onCreate 方法中初始化 SDK：

```java
public class MyApplication extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        
        // 初始化 Dehaze SDK
        DehazeSDK.initialize(new DehazeSDK.Builder()
                .setBaseUrl("https://api.dehaze.com/")
                .setDebug(BuildConfig.DEBUG));
    }
}
```

## 开发文档

各模块详细文档位于 [doc](doc) 目录下，现已按类别整理:

### 需求分析文档

- [登录模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/登录模块分析.md)
- [数据集模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/数据集模块分析.md)
- [系统管理模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/系统管理模块分析.md)
- [算法模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/算法模块分析.md)
- [对比模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/对比模块分析.md)
- [仪表盘模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/仪表盘模块分析.md)
- [评估模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/评估模块分析.md)
- [展示模块分析文档](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/需求分析/展示模块分析.md)

### 其他文档类别

- 架构设计文档请查看 [架构设计](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/架构设计) 目录
- 兼容性测试文档请查看 [兼容性测试](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/兼容性测试) 目录
- 单元测试文档请查看 [单元测试](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/单元测试) 目录

详细文档结构请参考 [文档目录说明](../dehaze-doc/docs/项目文档/图像去雾系统/代码详细设计/android/README.md)。

## 项目成果

### 功能完整性
完整迁移了 Web 端的所有核心功能：
- 用户认证、数据集管理、图像上传、去雾处理
- 结果对比、指标评估等核心业务
- 完善的系统管理功能

### 体验合规性
- 遵循 Android Material Design 3 与 Google 人机交互指南
- 重构了桌面端交互（如拖拽 → 滑动切换，画布 → 图片预览）
- 充分利用了 Android 原生能力

### 架构现代化
- 采用 Android Jetpack 架构组件实现清晰分层
- 使用 Navigation、Room、DataStore、ViewModel + LiveData 等现代化组件
- 实现了良好的代码组织和模块划分

## 后续建议

1. **持续集成/持续部署(CI/CD)**
   - 设置自动化构建和测试流程
   - 集成代码质量检查工具

2. **性能优化**
   - 进一步优化图片加载和处理性能
   - 减少内存占用和电池消耗

3. **功能扩展**
   - 增加更多去雾算法支持
   - 添加社交分享功能
   - 实现离线处理能力

4. **用户体验优化**
   - 增加夜间模式支持
   - 优化动画效果和过渡体验
   - 提供更多个性化设置选项

## 项目完成报告

查看完整的项目完成报告: [项目完成报告](PROJECT_COMPLETION_NOTICE.md)
