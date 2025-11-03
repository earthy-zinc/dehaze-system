# 第2章：Gradle环境搭建与基础配置

## 2.1 环境准备

### 2.1.1 JDK版本要求与配置

**Gradle JDK兼容性要求：**

Gradle 8.x对JDK版本有着明确的要求，这对于从Maven迁移的开发者来说需要特别注意：

| Gradle版本 | 最低JDK版本 | 推荐JDK版本 | 说明 |
|------------|-------------|-------------|------|
| 8.0 - 8.4 | JDK 8 | JDK 11/17 | 支持LTS版本 |
| 8.5+ | JDK 8 | JDK 17/21 | 推荐使用最新LTS |
| 9.0+ | JDK 17 | JDK 17/21 | 仅支持JDK 17+ |

**JDK安装和配置：**

```bash
# Windows环境配置示例
# 1. 下载并安装JDK 17（推荐使用Oracle JDK或OpenJDK）
# 2. 配置环境变量
set JAVA_HOME=C:\Program Files\Java\jdk-17.0.8
set PATH=%JAVA_HOME%\bin;%PATH%

# 3. 验证安装
java -version
javac -version
```

**多版本JDK管理策略：**

```bash
# 使用SDKMAN管理多版本JDK（Linux/Mac）
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# 安装多个JDK版本
sdk install java 8.0.382-tem
sdk install java 11.0.20-tem
sdk install java 17.0.8-tem
sdk install java 21.0.0-tem

# 切换JDK版本
sdk use java 17.0.8-tem

# Windows环境使用jenv或手动配置
# 或使用Chocolatey
choco install openjdk --version=17.0.8
```

**Gradle与JDK版本检测：**

```bash
# Gradle会自动检测并报告JDK版本信息
./gradlew --version

# 输出示例：
# Gradle 8.5
# Build time:   2023-11-02 14:38:15 UTC
# Revision:     2c4cc3070641cb3641b319d8206e6d9ad5b39b6d
# Kotlin:       1.9.10
# Groovy:       3.0.17
# Ant:          Apache Ant(TM) version 1.10.13 compiled on January 11 2023
# JVM:          17.0.8 (Eclipse Adoptium 17.0.8+7)
# OS:           Windows 10 10.0 amd64
```

### 2.1.2 Gradle Distribution选择

**三种Distribution类型对比：**

| Distribution类型 | 包含内容 | 适用场景 | 大小 | 下载速度 |
|------------------|----------|----------|------|-----------|
| Binary-only | 可执行文件、依赖库 | 仅运行构建 | ~150MB | 快 |
| Complete | Binary + 源码、文档 | 开发和调试 | ~200MB | 中等 |
| Source | 源代码 | 自定义编译 | ~50MB | 慢 |

**推荐选择策略：**

```bash
# 大多数情况选择Binary-only
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-bin.zip

# 开发Gradle插件或需要调试时选择Complete
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-all.zip

# 特殊需求下选择Source
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-src.zip
```

### 2.1.3 环境变量配置详解

**Windows环境配置：**

```cmd
# 1. 设置GRADLE_HOME
set GRADLE_HOME=C:\gradle\gradle-8.5

# 2. 添加到PATH
set PATH=%GRADLE_HOME%\bin;%PATH%

# 3. 设置GRADLE_USER_HOME（可选，自定义用户目录）
set GRADLE_USER_HOME=C:\gradle-user-home

# 4. 配置JVM参数（可选）
set GRADLE_OPTS=-Xmx4g -XX:MaxMetaspaceSize=512m

# 5. 配置代理（如果需要）
set GRADLE_USER_HOME=C:\gradle-user-home
# 在.gradle/gradle.properties中配置代理
```

**Linux/Mac环境配置：**

```bash
# ~/.bashrc 或 ~/.zshrc
export GRADLE_HOME=/opt/gradle/gradle-8.5
export PATH=$GRADLE_HOME/bin:$PATH
export GRADLE_USER_HOME=$HOME/.gradle

# JVM优化参数
export GRADLE_OPTS="-Xmx4g -XX:MaxMetaspaceSize=512m -XX:+UseG1GC"

# 代理配置（如果需要）
export JAVA_OPTS="-Dhttp.proxyHost=proxy.company.com -Dhttp.proxyPort=8080"

# 重新加载配置
source ~/.bashrc
```

**验证环境配置：**

```bash
# 验证Gradle安装
gradle --version

# 验证环境变量
echo $GRADLE_HOME
echo $GRADLE_USER_HOME
echo $GRADLE_OPTS

# 验证JDK配置
gradle -v | grep JVM
```

### 2.1.4 IDE集成配置

**IntelliJ IDEA集成：**

```groovy
// build.gradle - IDEA插件配置
plugins {
    id 'java'
    id 'idea'
}

// IDEA项目配置
idea {
    project {
        jdkName = '17'
        languageLevel = '17'
        targetBytecodeVersion = JavaVersion.VERSION_17
    }

    module {
        // 生成源码目录
        sourceDirs += file('src/generated/java')
        testSourceDirs += file('src/generated/test/java')

        // 排除目录
        excludeDirs += file('build/generated/sources/annotationProcessor/java/main')
        excludeDirs += file('out')

        // 依赖范围
        scopes.COMPILE.plus += [configurations.annotationProcessor]
        scopes.TEST.COMPILE.plus += [configurations.testAnnotationProcessor]

        // 下载源码和文档
        downloadJavadoc = true
        downloadSources = true
    }
}

// 自动生成IDEA文件
task generateIdeaFiles {
    dependsOn 'ideaProject', 'ideaModule', 'ideaWorkspace'
}
```

**Eclipse集成：**

```groovy
// build.gradle - Eclipse插件配置
plugins {
    id 'java'
    id 'eclipse'
}

// Eclipse项目配置
eclipse {
    project {
        name = 'my-gradle-project'
        comment = 'Gradle to Eclipse migration project'

        // 项目性质
        natures 'org.springframework.ide.eclipse.core.springnature'
        natures 'org.eclipse.jdt.core.javanature'
    }

    classpath {
        // 下载源码和文档
        downloadSources = true
        downloadJavadoc = true

        // 容器配置
        containers 'org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/JavaSE-17'

        // 自定义输出目录
        defaultOutputDir = file('build/classes/java/main')
    }

    jdt {
        sourceCompatibility = 17
        targetCompatibility = 17
        javaRuntimeName = 'JavaSE-17'
    }
}
```

## 2.2 Gradle安装方式

### 2.2.1 手动安装步骤

**Windows手动安装：**

```powershell
# 1. 下载Gradle
# 访问 https://gradle.org/releases/ 下载对应版本
# 例如：gradle-8.5-bin.zip

# 2. 解压到指定目录
# 解压到 C:\gradle\gradle-8.5

# 3. 配置系统环境变量
# 右键"此电脑" -> 属性 -> 高级系统设置 -> 环境变量
# 系统变量：
# GRADLE_HOME = C:\gradle\gradle-8.5
# PATH = %GRADLE_HOME%\bin;%PATH%

# 4. 验证安装
gradle --version
```

**Linux手动安装：**

```bash
# 1. 下载和解压
cd /opt
sudo wget https://services.gradle.org/distributions/gradle-8.5-bin.zip
sudo unzip gradle-8.5-bin.zip
sudo rm gradle-8.5-bin.zip

# 2. 创建符号链接
sudo ln -s /opt/gradle-8.5 /opt/gradle

# 3. 配置环境变量
echo 'export GRADLE_HOME=/opt/gradle' >> ~/.bashrc
echo 'export PATH=$GRADLE_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 验证安装
gradle --version
```

### 2.2.2 使用包管理器安装

**SDKMAN（推荐用于Linux/Mac）：**

```bash
# 安装SDKMAN
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# 安装Gradle
sdk install gradle 8.5

# 切换版本
sdk use gradle 8.5

# 列出可用版本
sdk list gradle

# 设置默认版本
sdk default gradle 8.5

# 更新Gradle
sdk upgrade gradle
```

**Homebrew（Mac）：**

```bash
# 安装Gradle
brew install gradle

# 验证安装
gradle --version

# 升级Gradle
brew upgrade gradle

# 安装特定版本
brew install gradle@8.5
```

**Chocolatey（Windows）：**

```powershell
# 安装Chocolatey（如果尚未安装）
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 安装Gradle
choco install gradle

# 安装特定版本
choco install gradle --version=8.5.0

# 升级Gradle
choco upgrade gradle
```

### 2.2.3 Gradle Wrapper最佳实践

**什么是Gradle Wrapper：**

Gradle Wrapper是Gradle的一个脚本，它允许项目在没有预先安装Gradle的机器上构建。这是企业级开发的标准实践，确保所有开发者和CI环境使用相同版本的Gradle。

**生成Wrapper：**

```bash
# 在项目根目录执行
gradle wrapper

# 指定Gradle版本
gradle wrapper --gradle-version 8.5

# 指定Distribution类型
gradle wrapper --gradle-version 8.5 --distribution-type all

# 生成Kotlin DSL版本的wrapper
gradle wrapper --gradle-version 8.5 --distribution-type all --kotlin-dsl
```

**Wrapper文件结构：**

```
project-root/
├── gradlew              # Unix/Linux执行脚本
├── gradlew.bat          # Windows执行脚本
└── gradle/
    └── wrapper/
        ├── gradle-wrapper.jar    # Wrapper JAR
        └── gradle-wrapper.properties # 配置文件
```

**gradle-wrapper.properties配置：**

```properties
# 配置Gradle版本和下载地址
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.5-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists

# 企业内部镜像配置（可选）
distributionUrl=https\://mirrors.company.com/gradle/gradle-8.5-bin.zip

# 网络超时配置（可选）
systemProp.http.connectionTimeout=60000
systemProp.http.socketTimeout=60000
```

**使用Wrapper：**

```bash
# 使用Wrapper执行Gradle命令
./gradlew build

# Windows环境
gradlew.bat build

# 显示Wrapper信息
./gradlew wrapper --version

# 更新Wrapper版本
./gradlew wrapper --gradle-version 8.6
```

**企业级Wrapper配置：**

```bash
# 1. 配置内部镜像（gradle.properties）
systemProp.gradle.wrapperUser=admin
systemProp.gradle.wrapperPassword=secret
systemProp.gradle.wrapperBaseUrl=https://repo.company.com/gradle

# 2. 配置离线模式（适用于内网环境）
gradle.properties配置：
org.gradle.offline=true

# 3. 配置代理
systemProp.http.proxyHost=proxy.company.com
systemProp.http.proxyPort=8080
systemProp.https.proxyHost=proxy.company.com
systemProp.https.proxyPort=8080

# 4. 配置认证
systemProp.gradle.user.home=/home/user/.gradle
```

### 2.2.4 版本管理策略

**多版本管理方案：**

```bash
# 1. 使用SDKMAN管理版本
sdk list gradle
# 切换到不同版本用于不同项目
sdk use gradle 8.4
sdk use gradle 8.5

# 2. 项目级版本锁定
# 每个项目使用自己的Gradle版本（通过Wrapper）
./gradlew wrapper --gradle-version 8.5

# 3. 全局默认版本
sdk default gradle 8.5
```

**版本升级策略：**

```bash
# 1. 检查可更新版本
./gradlew dependencyUpdates
# 或使用versions插件
./gradlew useLatestVersions

# 2. 渐进式升级
# 先升级patch版本
./gradlew wrapper --gradle-version 8.5.1

# 3. 测试兼容性
./gradlew clean build test

# 4. 升级minor版本
./gradlew wrapper --gradle-version 8.6

# 5. 升级major版本（需要仔细测试）
./gradlew wrapper --gradle-version 9.0
```

## 2.3 第一个Gradle项目

### 2.3.1 项目初始化（gradle init命令详解）

**gradle init命令参数：**

```bash
# 基本语法
gradle init --type <type> --dsl <language> --test-framework <framework> --package <package> --project-name <name>

# 参数说明
--type: basic, application, library, plugin
--dsl: groovy, kotlin
--test-framework: junit, testng, spock
--package: Java包名
--project-name: 项目名称
```

**创建Java应用项目：**

```bash
# 交互式创建（推荐初次使用）
gradle init

# 非交互式创建
gradle init --type java-application --dsl groovy --test-framework junit-jupiter --package com.example.demo --project-name my-demo-app

# Kotlin DSL版本
gradle init --type java-application --dsl kotlin --test-framework junit-jupiter --package com.example.demo --project-name my-demo-app
```

**交互式初始化示例：**

```bash
$ gradle init

Select type of project to generate:
  1: basic
  2: application
  3: library
  4: Gradle plugin
Enter selection (default: basic) [1..4] 2

Select implementation language:
  1: Java
  2: Kotlin
  3: Groovy
  4: Scala
Enter selection (default: Java) [1..4] 1

Select build script DSL:
  1: Groovy
  2: Kotlin
Enter selection (default: Groovy) [1..2] 1

Select test framework:
  1: JUnit 4
  2: TestNG
  3: Spock
  4: JUnit Jupiter
Enter selection (default: JUnit Jupiter) [1..4] 4

Project name (default: gradle-demo): my-first-gradle-app

Source package (default: my.first.gradle.app): com.example.demo

Generate build using new APIs and behavior (some features may change in the future)? (default: no) [yes, no] yes

> Task :init
BUILD SUCCESSFUL in 1m 2s
2 actionable tasks: 2 executed
```

**生成的项目结构：**

```
my-first-gradle-app/
├── build.gradle                 # 主构建脚本
├── settings.gradle             # 项目设置
├── gradle/
│   └── wrapper/               # Gradle Wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradlew                    # Unix/Linux执行脚本
├── gradlew.bat               # Windows执行脚本
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/demo/
│   │   │       └── App.java
│   │   └── resources/
│   └── test/
│       ├── java/
│       │   └── com/example/demo/
│       │       └── AppTest.java
│       └── resources/
├── .gitignore
└── README.md
```

### 2.3.2 项目结构解析

**Maven vs Gradle项目结构对比：**

```
Maven结构                    Gradle结构                     说明
src/main/java             src/main/java                 ✅ 相同
src/main/resources        src/main/resources            ✅ 相同
src/test/java             src/test/java                 ✅ 相同
src/test/resources        src/test/resources            ✅ 相同
src/main/webapp           src/main/webapp               ✅ 相同（Web项目）
src/main/filters          src/main/filters              ❌ Gradle使用不同方式
target/                   build/                        ✅ 构建输出目录
pom.xml                   build.gradle                  ❌ 构建脚本格式不同
settings.xml              gradle.properties             ❌ 配置文件不同
```

**Gradle特有的目录结构：**

```
project-root/
├── build/                   # 构建输出目录
│   ├── classes/            # 编译后的类文件
│   ├── resources/          # 处理后的资源文件
│   ├── libs/               # 打包后的JAR/WAR文件
│   ├── reports/            # 测试报告、代码质量报告等
│   ├── test-results/       # 测试结果
│   └── tmp/               # 临时文件
├── .gradle/               # Gradle缓存和工作目录
├── buildSrc/              # 共享构建逻辑（可选）
│   ├── src/main/java/     # 自定义插件和任务
│   └── build.gradle       # buildSrc的构建脚本
└── gradle/                # Gradle相关配置
    └── wrapper/           # Gradle Wrapper文件
```

### 2.3.3 构建脚本基础语法

**Groovy DSL基础语法：**

```groovy
// build.gradle - 基础配置

// 插件应用
plugins {
    id 'java'                    // Java项目插件
    id 'application'             // 应用程序插件
}

// 项目基本信息
group = 'com.example.demo'       // 等同于Maven的groupId
version = '1.0.0'               // 等同于Maven的version

// Java配置
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

// 仓库配置
repositories {
    mavenCentral()               // 等同于Maven的central仓库
    // mavenLocal()              // 本地Maven仓库
    // maven { url 'https://repo.spring.io/milestone' }
}

// 依赖管理
dependencies {
    // implementation：编译时需要，运行时需要（类似于Maven的compile）
    implementation 'org.apache.commons:commons-lang3:3.13.0'

    // api：编译时需要，运行时需要，并且会传递给依赖方
    api 'com.google.guava:guava:32.1.3-jre'

    // compileOnly：仅编译时需要（类似于Maven的provided）
    compileOnly 'javax.servlet:javax.servlet-api:4.0.1'

    // runtimeOnly：仅运行时需要（类似于Maven的runtime）
    runtimeOnly 'mysql:mysql-connector-java:8.2.0'

    // testImplementation：测试编译时需要（类似于Maven的test）
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
    testImplementation 'org.mockito:mockito-core:5.6.0'
}

// 应用程序配置
application {
    mainClass.set('com.example.demo.App')    // 主类
}

// 任务配置
test {
    useJUnitPlatform()                      // 使用JUnit 5
    testLogging {
        events "passed", "skipped", "failed"  // 测试日志级别
    }
}

// 自定义任务
task hello {
    doLast {
        println 'Hello, Gradle!'
    }
}

// 任务依赖关系
task dist(dependsOn: ['build', 'test']) {
    doLast {
        println 'Distribution completed!'
    }
}
```

**Kotlin DSL基础语法：**

```kotlin
// build.gradle.kts - Kotlin DSL配置

plugins {
    java
    application
}

group = "com.example.demo"
version = "1.0.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.apache.commons:commons-lang3:3.13.0")
    api("com.google.guava:guava:32.1.3-jre")
    compileOnly("javax.servlet:javax.servlet-api:4.0.1")
    runtimeOnly("mysql:mysql-connector-java:8.2.0")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testImplementation("org.mockito:mockito-core:5.6.0")
}

application {
    mainClass.set("com.example.demo.App")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
    }
}

// 自定义任务
tasks.register("hello") {
    doLast {
        println("Hello, Gradle with Kotlin DSL!")
    }
}

// 任务依赖
tasks.register("dist") {
    dependsOn(tasks.build, tasks.test)
    doLast {
        println("Distribution completed!")
    }
}
```

### 2.3.4 常用命令介绍

**基础命令：**

```bash
# 构建项目
./gradlew build

# 清理构建产物
./gradlew clean

# 运行测试
./gradlew test

# 运行应用程序
./gradlew run

# 打包应用
./gradlew assemble

# 生成项目报告
./gradlew projects
./gradlew tasks
./gradlew dependencies

# 显示项目属性
./gradlew properties

# 帮助命令
./gradlew help
./gradlew help --task build
```

**高级命令：**

```bash
# 并行构建
./gradlew build --parallel

# 离线模式构建
./gradlew build --offline

# 重新构建所有
./gradlew build --rerun-tasks

# 持续模式（文件变化时自动构建）
./gradlew build --continuous

# 构建缓存配置
./gradlew build --build-cache

# 生成构建扫描
./gradlew build --scan

# 显示任务依赖图
./gradlew build --dependency-graph

# 指定任务执行顺序
./gradlew clean build test

# 运行特定任务
./gradlew :subproject-name:build

# 运行测试并生成报告
./gradlew test --continue --tests "*Test"
```

## 2.4 Gradle配置文件详解

### 2.4.1 settings.gradle文件作用与配置

**settings.gradle主要作用：**

1. **声明项目结构**：定义多模块项目的子项目
2. **项目命名**：设置项目名称和属性
3. **仓库配置**：全局仓库配置
4. **插件管理**：插件版本管理
5. **构建脚本配置**：构建脚本仓库和依赖

**基础配置示例：**

```groovy
// settings.gradle

// 项目名称
rootProject.name = 'multi-module-project'

// 包含子项目
include 'common:utils'
include 'common:domain'
include 'infrastructure:persistence'
include 'application:web'
include 'application:batch'

// 项目重命名
project(':common:utils').name = 'common-utils'
project(':infrastructure:persistence').name = 'persistence-module'

// 设置项目目录
include 'external-service'
project(':external-service').projectDir = file('../external-service-project')

// 条件化包含子项目
if (file('modules/admin').exists()) {
    include 'modules:admin'
}
```

**高级配置示例：**

```groovy
// settings.gradle - 高级配置

// 1. 插件管理
pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        maven { url 'https://repo.spring.io/milestone' }
    }

    // 插件版本声明
    plugins {
        id 'org.springframework.boot' version '3.2.0'
        id 'io.spring.dependency-management' version '1.1.4'
        id 'com.github.johnrengelman.shadow' version '8.1.1'
    }
}

// 2. 依赖解析管理
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
        maven { url 'https://repo.spring.io/milestone' }
        maven { url 'https://repo.spring.io/snapshot' }
    }
}

// 3. 构建缓存配置
buildCache {
    local {
        enabled = true
        directory = new File(rootDir, ".gradle-build-cache")
    }

    remote(HttpBuildCache) {
        enabled = System.getenv('CI') != null
        url = 'https://gradle-build-cache.company.com/cache'
        credentials {
            username = System.getenv('GRADLE_CACHE_USERNAME')
            password = System.getenv('GRADLE_CACHE_PASSWORD')
        }
    }
}

// 4. 项目初始化钩子
gradle.projectsLoaded {
    println "Loading ${gradle.rootProject.name}..."
}

gradle.beforeProject { project ->
    if (project.name.startsWith('common')) {
        project.ext.isCommonModule = true
    }
}
```

### 2.4.2 build.gradle核心配置

**项目级build.gradle配置：**

```groovy
// build.gradle - 根项目配置

// 1. 插件应用
plugins {
    id 'java-platform'        // Java平台插件（用于BOM管理）
    id 'io.spring.dependency-management' version '1.1.4' apply false
    id 'org.springframework.boot' version '3.2.0' apply false
}

// 2. 子项目通用配置
subprojects {
    // 应用基础插件
    apply plugin: 'java'
    apply plugin: 'java-library'

    // Java配置
    java {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17

        // 生成源码和文档JAR
        withSourcesJar()
        withJavadocJar()
    }

    // 仓库配置
    repositories {
        mavenCentral()
    }

    // 依赖管理
    dependencies {
        // 统一的测试依赖
        testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
        testImplementation 'org.mockito:mockito-core:5.6.0'
        testImplementation 'org.assertj:assertj-core:3.24.2'
    }

    // 测试配置
    test {
        useJUnitPlatform()

        // JVM参数
        jvmArgs = [
            '-Dspring.profiles.active=test',
            '-Dfile.encoding=UTF-8'
        ]

        // 系统属性
        systemProperty 'java.awt.headless', 'true'
        systemProperty 'test.groups', System.getProperty('test.groups', '')

        // 测试日志
        testLogging {
            events "passed", "skipped", "failed"
            exceptionFormat "full"
        }

        // 并行测试
        maxParallelForks = Runtime.runtime.availableProcessors().intdiv(2) ?: 1
    }

    // 编译配置
    tasks.withType(JavaCompile).configureEach {
        options.encoding = 'UTF-8'
        options.compilerArgs += [
            '-Xlint:unchecked',
            '-Xlint:deprecation',
            '-parameters'
        ]
    }
}

// 3. 特定项目配置
project(':common:utils') {
    dependencies {
        api 'org.apache.commons:commons-lang3:3.13.0'
        api 'org.apache.commons:commons-collections4:4.4'
    }
}

project(':application:web') {
    apply plugin: 'org.springframework.boot'
    apply plugin: 'io.spring.dependency-management'

    dependencies {
        implementation project(':common:domain')
        implementation 'org.springframework.boot:spring-boot-starter-web'
        developmentOnly 'org.springframework.boot:spring-boot-devtools'
    }
}
```

### 2.4.3 gradle.properties全局属性

**gradle.properties配置示例：**

```properties
# =============================================================================
# Gradle全局属性配置
# =============================================================================

# ------------------------------
# 项目信息
# ------------------------------
group=com.example.multi-module
version=1.0.0-SNAPSHOT
description=Multi-module Spring Boot application

# ------------------------------
# JVM配置
# ------------------------------
# 内存配置
org.gradle.jvmargs=-Xmx4g -XX:MaxMetaspaceSize=512m -XX:+UseG1GC

# Java版本
org.gradle.java.home=/Library/Java/JavaVirtualMachines/jdk-17.jdk/Contents/Home

# ------------------------------
# 构建优化配置
# ------------------------------
# 并行构建
org.gradle.parallel=true

# 配置缓存
org.gradle.configuration-cache=true
org.gradle.configuration-cache.problems=warn

# 构建缓存
org.gradle.caching=true

# 按需配置
org.gradle.configureondemand=true

# ------------------------------
# 网络配置
# ------------------------------
# 代理设置（如果需要）
#systemProp.http.proxyHost=proxy.company.com
#systemProp.http.proxyPort=8080
#systemProp.https.proxyHost=proxy.company.com
#systemProp.https.proxyPort=8080

# 超时设置
systemProp.http.connectionTimeout=60000
systemProp.http.socketTimeout=60000

# ------------------------------
# 仓库配置
# ------------------------------
# 镜像配置
systemProp.gradle.wrapperUser=admin
systemProp.gradle.wrapperPassword=secret
systemProp.gradle.wrapperBaseUrl=https://repo.company.com

# Maven仓库镜像
#systemProp.maven.repo.local=/home/user/.m2/repository

# ------------------------------
# 开发配置
# ------------------------------
# 开发模式
devMode=true

# 环境配置
env=development

# 跳过测试（开发时可使用）
#skipTests=true

# ------------------------------
# 版本管理
# ------------------------------
# Spring Boot版本
springBootVersion=3.2.0

# 依赖版本库
versions.java=17
versions.kotlin=1.9.10
versions.junit=5.10.0
versions.mockito=5.6.0

# ------------------------------
# 性能监控
# ------------------------------
# 构建扫描
org.gradle.buildScan.termsOfServiceUrl=https://gradle.com/terms-of-service
org.gradle.buildScan.termsOfServiceAgree=yes

# 调试模式
#org.gradle.debug=true
```

### 2.4.4 配置文件优先级

**Gradle配置文件优先级顺序：**

1. **命令行参数**（最高优先级）
2. **项目gradle.properties**
3. **用户主目录gradle.properties**
4. **系统环境变量**
5. **默认值**（最低优先级）

**配置文件查找顺序：**

```
项目目录/.gradle/gradle.properties    # 项目特定配置
用户主目录/.gradle/gradle.properties   # 用户全局配置
系统配置目录/gradle.properties          # 系统级配置
```

**配置验证和调试：**

```bash
# 显示当前配置
./gradlew properties

# 显示特定属性
./gradlew properties --property group
./gradlew properties --property version

# 显示任务图
./gradlew tasks --all

# 显示项目结构
./gradlew projects

# 调试模式
./gradlew build --debug --stacktrace

# 显示依赖解析
./gradlew dependencies --configuration runtimeClasspath
```

## 2.5 IDE深度集成

### 2.5.1 IntelliJ IDEA集成配置

**项目导入和配置：**

```groovy
// build.gradle - IDEA配置
plugins {
    id 'java'
    id 'idea'
    id 'org.springframework.boot' version '3.2.0'
}

// IDEA项目配置
idea {
    project {
        // 项目JDK设置
        jdkName = '17'
        languageLevel = '17'
        targetBytecodeVersion = JavaVersion.VERSION_17

        // 项目编码
        encoding = 'UTF-8'

        // VCS配置
        vcs = 'Git'

        // 输出路径
        outputDir = file('build/out')

        // 编译器输出
        compiler {
            outputFile = file('build/idea-compiler-output.xml')
        }
    }

    module {
        // 源码目录
        sourceDirs += file('src/generated/java')
        sourceDirs += file('src/integration-test/java')

        // 测试源码目录
        testSourceDirs += file('src/integration-test/java')
        testSourceDirs += file('src/generated/test/java')

        // 资源目录
        resourceDirs += file('src/main/resources')
        resourceDirs += file('src/test/resources')

        // 排除目录
        excludeDirs += file('build')
        excludeDirs += file('out')
        excludeDirs += file('.gradle')
        excludeDirs += file('node_modules')

        // 下载源码和文档
        downloadJavadoc = true
        downloadSources = true

        // 内容根目录
        inheritOutputDirs = false
        outputDir = file('build/classes/java/main')
        testOutputDir = file('build/classes/java/test')

        // 依赖范围
        scopes.COMPILE.plus += [configurations.annotationProcessor]
        scopes.TEST.COMPILE.plus += [configurations.testAnnotationProcessor]
        scopes.RUNTIME.plus += [configurations.runtimeOnly]

        // 模块名称
        name = project.name
    }

    workspace {
        // 工作区配置
        iws {
            withXml { xml ->
                def project = xml.node.component.find { it.@name == 'ProjectRootManager' }
                project.@languageLevel = 'JDK_17'
                project.@project-jdk-name = '17'
            }
        }
    }
}

// 自定义IDEA任务
task generateIdeaFiles {
    group = 'IDE'
    description = 'Generate IntelliJ IDEA project files'

    dependsOn 'ideaProject', 'ideaModule', 'ideaWorkspace'
    doLast {
        println "IntelliJ IDEA files generated successfully"
    }
}

task cleanIdeaFiles(type: Delete) {
    group = 'IDE'
    description = 'Clean IntelliJ IDEA project files'

    delete '.idea', '*.iml', '*.ipr', '*.iws'
}
```

**Spring Boot IDEA配置：**

```groovy
// Spring Boot项目特定配置
idea {
    module {
        // Spring Boot配置
        inheritOutputDirs = false
        outputDir = file('build/classes/java/main')
        testOutputDir = file('build/classes/java/test')

        // Spring Boot资源目录
        resourceDirs += file('src/main/resources')

        // 排除不必要的目录
        excludeDirs += file('build/classes/java/main/META-INF')
        excludeDirs += file('build/classes/java/test/META-INF')

        // Spring Boot DevTools
        if (project.hasProperty('dev')) {
            scopes.COMPILE.plus += configurations.developmentOnly
        }
    }

    project {
        // Spring Boot运行配置
        vcs = 'Git'

        // 编译器输出路径
        compiler.outputDir = file('build/idea-output')
    }
}

// Spring Boot运行配置生成
task createRunConfigurations {
    doLast {
        def runConfigDir = file('.idea/runConfigurations')
        runConfigDir.mkdirs()

        // 主应用运行配置
        def mainAppConfig = """
            <component name="ProjectRunConfigurationManager">
              <configuration default="false" name="${project.name}-Application" type="SpringBootApplicationConfigurationType" factoryName="Spring Boot">
                <module name="${project.name}" />
                <option name="SPRING_BOOT_MAIN_CLASS" value="${bootJar.mainClass.get()}" />
                <option name="ALTERNATIVE_JRE_PATH" />
                <option name="SHORTEN_COMMAND_LINE" value="NONE" />
                <option name="ENABLE_ALTERNATIVE_JRE" value="false" />
                <option name="PROGRAM_PARAMETERS" value="" />
                <option name="VM_PARAMETERS" value="-Dspring.profiles.active=dev" />
                <option name="WORKING_DIRECTORY" value="\$PROJECT_DIR\$" />
                <option name="INCLUDE_PROVIDED_SCOPE" value="true" />
                <option name="RUN_TARGET_PROJECT_NAME" value="${project.name}" />
                <option name="RUN_TARGET_SELECTION_MODE" value="CURRENT_FILE" />
                <envs />
                <method v="2">
                  <option name="Make" enabled="true" />
                </method>
              </configuration>
            </component>
        """.stripIndent()

        file("$runConfigDir/${project.name}-Application.xml").text = mainAppConfig
    }
}
```

### 2.5.2 Eclipse集成配置

**Eclipse项目配置：**

```groovy
// build.gradle - Eclipse配置
plugins {
    id 'java'
    id 'eclipse'
    id 'eclipse-wtp'  // Web项目支持
}

// Eclipse项目配置
eclipse {
    project {
        name = project.name
        comment = "${project.description ?: project.name} - Gradle project"

        // 项目性质
        natures = [
            'org.eclipse.jdt.core.javanature',
            'org.eclipse.wst.common.project.facet.core.nature'
        ]

        // 构建命令
        buildCommands = [
            'org.eclipse.jdt.core.javabuilder',
            'org.eclipse.wst.common.project.facet.core.builder'
        ]

        // 链接资源
        linkedResources = [
            new Link(name: 'gradle', type: 2, location: projectDir.absolutePath + '/.gradle'),
            new Link(name: 'build', type: 2, location: projectDir.absolutePath + '/build')
        ]
    }

    classpath {
        // 下载源码和文档
        downloadSources = true
        downloadJavadoc = true

        // 容器配置
        containers = [
            'org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/JavaSE-17',
            'org.eclipse.jst.j2ee.internal.web.container',
            'org.eclipse.jst.j2ee.internal.module.container'
        ]

        // 自定义输出目录
        defaultOutputDir = file('build/classes/java/main')

        // 依赖配置
        file {
            whenMerged { classpath ->
                // 移除重复的容器
                classpath.entries.removeAll {
                    it.kind == 'con' && it.path.startsWith('org.eclipse.jdt.USER_LIBRARY')
                }

                // 配置测试输出目录
                classpath.entries.each { entry ->
                    if (entry.kind == 'src' && entry.path.startsWith('src/test/')) {
                        entry.output = 'build/classes/java/test'
                    }
                }
            }
        }
    }

    wtp {
        // Web项目配置
        facet {
            facet name: 'jst.java', version: '17'
            facet name: 'jst.web', version: '5.0'
            facet name: 'wst.jsdt.web', version: '1.0'
        }

        // 部署配置
        component {
            context path = "/${project.name}"
            name = project.name
        }

        // 模块配置
        module {
            sourceDirs += file('src/main/resources')
            sourceDirs += file('src/test/resources')
        }
    }

    jdt {
        // Java编译器设置
        sourceCompatibility = 17
        targetCompatibility = 17
        javaRuntimeName = 'JavaSE-17'

        // 编译器设置
        javaHome = file(System.getProperty('java.home'))
    }
}

// 自定义Eclipse任务
task generateEclipseFiles {
    group = 'IDE'
    description = 'Generate Eclipse project files'

    dependsOn 'eclipseProject', 'eclipseClasspath', 'eclipseJdt', 'eclipseWtpComponent', 'eclipseWtpFacet'
    doLast {
        println "Eclipse files generated successfully"
        println "Import the project into Eclipse using: File -> Import -> General -> Existing Projects into Workspace"
    }
}

task cleanEclipseFiles(type: Delete) {
    group = 'IDE'
    description = 'Clean Eclipse project files'

    delete '.project', '.classpath', '.settings', '.factorypath'
}
```

### 2.5.3 VS Code集成配置

**VS Code工作区配置：**

```json
// .vscode/settings.json
{
    "java.home": "${env:JAVA_HOME}",
    "java.configuration.updateBuildConfiguration": "automatic",
    "java.compile.nullAnalysis.mode": "automatic",
    "java.debug.settings.onBuildFailureProceed": true,
    "java.format.settings.url": ".vscode/java-format.xml",
    "java.saveActions.organizeImports": true,
    "java.completion.importOrder": [
        "java",
        "javax",
        "org",
        "com",
        ""
    ],
    "gradle.autoDetect": "on",
    "gradle.wrapper.enabled": true,
    "gradle.reuseTerminals": true,
    "gradle.nestedProjects": true,
    "files.exclude": {
        "**/.gradle": true,
        "**/build": true,
        "**/bin": true,
        "**/.classpath": true,
        "**/.project": true,
        "**/.settings": true,
        "**/*.iml": true
    },
    "files.watcherExclude": {
        "**/.gradle/**": true,
        "**/build/**": true
    },
    "search.exclude": {
        "**/.gradle/**": true,
        "**/build/**": true
    }
}
```

**VS Code任务配置：**

```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Gradle Build",
            "type": "gradle",
            "script": "build",
            "description": "Build the project using Gradle",
            "group": "build",
            "problemMatcher": [
                "$gradle"
            ]
        },
        {
            "label": "Gradle Test",
            "type": "gradle",
            "script": "test",
            "description": "Run tests using Gradle",
            "group": "test",
            "problemMatcher": [
                "$gradle"
            ]
        },
        {
            "label": "Gradle Boot Run",
            "type": "gradle",
            "script": "bootRun",
            "description": "Run Spring Boot application",
            "group": "build",
            "problemMatcher": [
                "$gradle"
            ]
        },
        {
            "label": "Gradle Clean",
            "type": "gradle",
            "script": "clean",
            "description": "Clean the project using Gradle",
            "group": "build"
        }
    ]
}
```

**VS Code启动配置：**

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "java",
            "name": "Launch Current File",
            "request": "launch",
            "mainClass": "${file}",
            "projectName": "${workspaceFolderBasename}"
        },
        {
            "type": "java",
            "name": "Launch App",
            "request": "launch",
            "mainClass": "com.example.demo.DemoApplication",
            "projectName": "${workspaceFolderBasename}",
            "args": "--spring.profiles.active=dev",
            "env": {
                "SPRING_PROFILES_ACTIVE": "dev"
            },
            "vmArgs": [
                "-Dspring.profiles.active=dev",
                "-Xmx2g",
                "-XX:+UseG1GC"
            ]
        },
        {
            "type": "java",
            "name": "Debug Tests",
            "request": "launch",
            "mainClass": "org.junit.platform.console.ConsoleLauncher",
            "args": [
                "--select-class",
                "${file}",
                "--details=summary"
            ],
            "projectName": "${workspaceFolderBasename}",
            "classPaths": [
                "${workspaceFolder}/build/classes/java/test",
                "${workspaceFolder}/build/resources/test"
            ]
        }
    ]
}
```

### 2.5.4 代码提示和调试配置

**Gradle DSL代码提示增强：**

```groovy
// build.gradle - DSL类型提示增强
import org.gradle.api.Project
import org.gradle.api.JavaVersion
import org.gradle.api.plugins.JavaPluginExtension
import org.gradle.api.tasks.JavaCompile
import org.gradle.api.tasks.testing.Test

// 类型安全的扩展配置
extensions.create('appConfig', AppConfigExtension)

class AppConfigExtension {
    String appName = 'My App'
    String version = '1.0.0'
    Map<String, Object> properties = [:]
}

// 类型安全的依赖管理
dependencies {
    // 使用类型安全的依赖声明
    implementation platform('org.springframework.boot:spring-boot-dependencies:3.2.0')
    implementation 'org.springframework.boot:spring-boot-starter-web'

    // 条件化依赖
    if (project.hasProperty('enableActuator')) {
        implementation 'org.springframework.boot:spring-boot-starter-actuator'
    }
}

// 类型安全的任务配置
tasks.named('compileJava', JavaCompile) {
    options.encoding = 'UTF-8'
    options.compilerArgs += ['-Xlint:unchecked', '-Xlint:deprecation']
    options.release.set(17)
}

tasks.named('test', Test) {
    useJUnitPlatform()
    maxHeapSize = '1g'
    testLogging {
        events 'passed', 'skipped', 'failed'
        exceptionFormat 'full'
    }
}
```

**调试配置增强：**

```groovy
// build.gradle - 调试配置

// 启用调试模式
if (project.hasProperty('debug')) {
    tasks.withType(Test).configureEach {
        jvmArgs = ['-Xdebug', '-Xrunjdwp:transport=dt_socket,server=y,suspend=y,address=5005']
    }
}

// 开发模式配置
if (project.hasProperty('devMode')) {
    tasks.withType(JavaCompile).configureEach {
        options.compilerArgs += ['-g']  // 生成调试信息
    }

    // Spring Boot开发工具
    dependencies {
        developmentOnly 'org.springframework.boot:spring-boot-devtools'
    }

    // 热部署配置
    bootRun {
        jvmArgs = [
            '-Dspring.devtools.restart.enabled=true',
            '-Dspring.devtools.livereload.enabled=true'
        ]
    }
}

// 性能分析配置
if (project.hasProperty('profile')) {
    tasks.withType(Test).configureEach {
        jvmArgs = [
            '-XX:+FlightRecorder',
            '-XX:StartFlightRecording=duration=60s,filename=build/test-profiling.jfr'
        ]
    }
}
```

---

## 本章总结

通过本章的学习，您应该已经掌握了Gradle环境的完整搭建过程，包括：

**核心技能掌握：**
1. ✅ 理解了Gradle与JDK的版本兼容性要求
2. ✅ 掌握了多种Gradle安装方式（手动、包管理器、Wrapper）
3. ✅ 学会了创建和配置第一个Gradle项目
4. ✅ 深入理解了Gradle配置文件的作用和配置方法
5. ✅ 掌握了主流IDE的深度集成配置

**关键实践要点：**
- **Wrapper是标准实践**：企业项目必须使用Gradle Wrapper确保构建一致性
- **IDE集成很重要**：良好的IDE配置能显著提升开发效率
- **环境配置要规范**：统一的JDK版本和配置参数是团队协作的基础
- **配置文件优先级要清楚**：理解不同配置文件的优先级避免配置冲突

**下一步行动：**
1. 在您的开发环境中完成Gradle安装和配置
2. 使用Gradle创建一个示例项目
3. 配置您常用的IDE以获得最佳开发体验
4. 继续学习下一章的核心概念详解

掌握了环境搭建和基础配置后，您已经为深入学习Gradle的核心概念打下了坚实基础。在下一章中，我们将详细解析Gradle的Project、Task、生命周期等核心概念，帮助您建立对Gradle构建机制的深入理解。