# dehaze-android

将 dehaze-front-vue 完整重写为原生 Android 应用，使用 Java 语言开发。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Android前端架构文档.md)。

## 技术栈

- Java + MVVM + Android Jetpack Components
- Retrofit + OkHttp（网络）、Glide（图片加载）
- Room（本地缓存）、DataStore（首选项）

## 环境要求

- Android Studio Flamingo 或更高版本
- JDK 8 或更高版本
- Android SDK API Level 23+
- Gradle 8.0+

## 快速开始

```bash
./gradlew build              # 构建
./gradlew installDebug       # 安装到设备
./gradlew testDebugUnitTest  # 运行单元测试
./gradlew jacocoTestReport   # 生成测试覆盖率报告
```

测试覆盖率报告位置: `app/build/reports/jacoco/jacocoTestReport/html/index.html`

## API 配置

应用默认连接本地开发服务器，地址配置在 [DehazeApplication.java](app/src/main/java/com/pei/dehaze/DehazeApplication.java) 中：

```java
DehazeSDK.Builder()
    .setBaseUrl("http://10.0.2.2:8989") // Android模拟器访问本机需要使用10.0.2.2
    .setDebug(true)
```

如果需要更改服务器地址，请修改此处配置。
