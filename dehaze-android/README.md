# 图像去雾系统 Android 应用

这是一个将 dehaze-front-vue（桌面 Web）完整重写为原生 Android 应用的项目，使用 Java 语言开发。

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

## 主要组件

- Navigation Component - 应用内页面导航
- ViewModel + LiveData - UI 状态管理
- RecyclerView - 列表展示
- ViewPager2 - 页面滑动切换
- DataBinding - 数据绑定
- CameraX - 拍照功能

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

应用默认连接本地开发服务器，地址配置在 [DehazeApplication.java](file:///e:/DehazeSystem/dehaze-android/app/src/main/java/com/pei/dehaze/DehazeApplication.java#L11-L18) 中:

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

## 开发文档

各模块详细文档位于 [doc](file:///e:/DehazeSystem/dehaze-android/doc) 目录下:

- [登录模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/login_module_analysis.md)
- [数据集模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/dataset_module_analysis.md)
- [系统管理模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/system_module_analysis.md)
- [算法模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/algorithm_module_analysis.md)
- [对比模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/compare_module_analysis.md)
- [仪表盘模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/dashboard_module_analysis.md)
- [评估模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/evaluation_module_analysis.md)
- [展示模块分析文档](file:///e:/DehazeSystem/dehaze-android/doc/presentation_module_analysis.md)

以及对应的架构、兼容性、测试文档等。

## 项目完成报告

查看完整的项目完成报告: [项目完成报告](file:///e:/DehazeSystem/dehaze-android/doc/project_completion_report.md)