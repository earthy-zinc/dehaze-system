# 第九章：Groovy在Android开发中的应用

> Groovy在Android开发中扮演着重要角色，特别是在Gradle构建系统和动态配置方面。了解Groovy在Android中的应用，能够帮助开发者更好地理解Android构建工具链，并提升开发效率。

## 9.1 Gradle在Android中的应用

### 9.1.1 Android Gradle插件基础

```groovy
// Android应用的 build.gradle 文件结构
plugins {
    id 'com.android.application'
    id 'kotlin-android'  // 如果使用Kotlin
    id 'kotlin-kapt'     // 如果使用注解处理器
}

android {
    // 基础配置
    compileSdkVersion 33
    buildToolsVersion "33.0.0"

    defaultConfig {
        applicationId "com.example.myapp"
        minSdkVersion 21
        targetSdkVersion 33
        versionCode 1
        versionName "1.0.0"

        // 测试配置
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"

        // 构建配置字段
        buildConfigField "String", "API_BASE_URL", '"https://api.example.com"'
        buildConfigField "boolean", "DEBUG_MODE", "true"
        buildConfigField "long", "BUILD_TIMESTAMP", "${System.currentTimeMillis()}L"

        // Manifest占位符
        manifestPlaceholders = [
            appName: "My App",
            appIcon: "@mipmap/ic_launcher"
        ]
    }

    // 签名配置
    signingConfigs {
        debug {
            storeFile file('keystore/debug.keystore')
            storePassword 'android'
            keyAlias 'androiddebugkey'
            keyPassword 'android'
        }

        release {
            storeFile file('keystore/release.keystore')
            storePassword System.getenv('KEYSTORE_PASSWORD')
            keyAlias System.getenv('KEY_ALIAS')
            keyPassword System.getenv('KEY_PASSWORD')
        }
    }

    // 构建类型
    buildTypes {
        debug {
            minifyEnabled false
            debuggable true
            applicationIdSuffix ".debug"
            versionNameSuffix "-debug"

            buildConfigField "boolean", "DEBUG_MODE", "true"
            buildConfigField "String", "API_BASE_URL", '"https://api-dev.example.com"'

            manifestPlaceholders = [
                appName: "My App (Debug)",
                appIcon: "@mipmap/ic_launcher_debug"
            ]
        }

        release {
            minifyEnabled true
            debuggable false
            signingConfig signingConfigs.release

            buildConfigField "boolean", "DEBUG_MODE", "false"
            buildConfigField "String", "API_BASE_URL", '"https://api.example.com"'

            // ProGuard配置
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'

            manifestPlaceholders = [
                appName: "My App",
                appIcon: "@mipmap/ic_launcher"
            ]
        }

        // 自定义构建类型
        staging {
            initWith release
            applicationIdSuffix ".staging"
            versionNameSuffix "-staging"
            debuggable true

            buildConfigField "String", "API_BASE_URL", '"https://api-staging.example.com"'

            signingConfig signingConfigs.debug
        }
    }

    // 产品风味
    flavorDimensions "version"

    productFlavors {
        free {
            dimension "version"
            applicationIdSuffix ".free"
            versionNameSuffix "-free"

            buildConfigField "boolean", "PREMIUM_FEATURES", "false"
        }

        paid {
            dimension "version"
            applicationIdSuffix ".paid"
            versionNameSuffix "-paid"

            buildConfigField "boolean", "PREMIUM_FEATURES", "true"
        }

        enterprise {
            dimension "version"
            applicationIdSuffix ".enterprise"
            versionNameSuffix "-enterprise"

            buildConfigField "boolean", "PREMIUM_FEATURES", "true"
            buildConfigField "boolean", "ENTERPRISE_MODE", "true"
        }
    }

    // 变体过滤
    variantFilter { variant ->
        def names = variant.flavors*.name
        if (names.contains("enterprise") && variant.buildType.name == "debug") {
            // 排除enterprise的debug版本
            variant.setIgnore(true)
        }
    }

    // 编译选项
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }

    // Kotlin编译选项（如果使用Kotlin）
    kotlinOptions {
        jvmTarget = '1.8'
        freeCompilerArgs += ['-Xjvm-default=enable']
    }

    // 数据绑定
    buildFeatures {
        dataBinding true
        viewBinding true
    }

    // 资源配置
    resourcePrefix 'app_'

    // Lint选项
    lintOptions {
        abortOnError false
        checkReleaseBuilds false
        disable 'MissingTranslation'
    }

    // 测试选项
    testOptions {
        unitTests {
            includeAndroidResources = true
        }
    }

    // 分包配置
    dexOptions {
        javaMaxHeapSize "4g"
        preDexLibraries = true
    }

    // NDK配置
    ndkVersion "23.1.7779620"

    externalNativeBuild {
        cmake {
            path 'src/main/cpp/CMakeLists.txt'
            version '3.18.1'
        }
    }
}

// 依赖配置
dependencies {
    // Android核心库
    implementation 'androidx.core:core-ktx:1.9.0'
    implementation 'androidx.appcompat:appcompat:1.5.4'
    implementation 'com.google.android.material:material:1.6.1'
    implementation 'androidx.constraintlayout:constraintlayout:2.1.4'

    // 测试依赖
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.3'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'

    // 网络库
    implementation 'com.squareup.retrofit2:retrofit:2.9.0'
    implementation 'com.squareup.retrofit2:converter-gson:2.9.0'
    implementation 'com.squareup.okhttp3:logging-interceptor:4.10.0'

    // 图片加载
    implementation 'com.github.bumptech.glide:glide:4.14.2'

    // 依赖注入
    implementation 'com.google.dagger:hilt-android:2.44'
    kapt 'com.google.dagger:hilt-compiler:2.44'

    // 分页库
    implementation 'androidx.paging:paging-runtime:3.1.1'

    // 根据产品风味添加不同依赖
    freeImplementation 'com.google.android.gms:play-services-ads:21.3.0'
    paidImplementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.5.1'
    enterpriseImplementation 'com.squareup.okhttp3:okhttp:4.10.0'
}
```

### 9.1.2 动态版本和配置管理

```groovy
// 版本管理文件 (gradle.properties)
gradle.properties
```

```properties
# 版本号配置
VERSION_NAME=1.2.0
VERSION_CODE=3

# 编译版本
COMPILE_SDK=33
TARGET_SDK=33
MIN_SDK=21

# 构建工具版本
BUILD_TOOLS=33.0.0

# 依赖版本
KOTLIN_VERSION=1.7.20
ANDROIDX_CORE_VERSION=1.9.0
MATERIAL_VERSION=1.6.1
RETROFIT_VERSION=2.9.0

# 签名配置
KEYSTORE_FILE=keystore/release.keystore
KEY_ALIAS=myapp

# 功能开关
ENABLE_CRASHLYTICS=true
ENABLE_ANALYTICS=false
```

```groovy
// build.gradle (Project level)
// 使用properties文件中的版本
ext {
    kotlin_version = project.hasProperty('KOTLIN_VERSION') ? project.KOTLIN_VERSION : '1.7.20'
    compileSdkVersion = project.hasProperty('COMPILE_SDK') ? project.COMPILE_SDK.toInteger() : 33
    targetSdkVersion = project.hasProperty('TARGET_SDK') ? project.TARGET_SDK.toInteger() : 33
    minSdkVersion = project.hasProperty('MIN_SDK') ? project.MIN_SDK.toInteger() : 21

    // 动态版本计算
    versionCode = project.hasProperty('VERSION_CODE') ? project.VERSION_CODE.toInteger() : 1
    versionName = project.hasProperty('VERSION_NAME') ? project.VERSION_NAME : '1.0.0'

    // 环境配置
    isCiBuild = System.getenv('CI') == 'true'
    isReleaseBuild = project.gradle.startParameter.taskNames.any { it.toLowerCase().contains('release') }

    // 时间戳
    buildTime = new Date().format('yyyy-MM-dd HH:mm:ss')
    buildTimestamp = System.currentTimeMillis()

    // Git信息
    gitCommit = 'git rev-parse --short HEAD'.execute().text.trim()
    gitBranch = 'git rev-parse --abbrev-ref HEAD'.execute().text.trim()
    gitCommitCount = 'git rev-list --count HEAD'.execute().text.trim().toInteger()
}

allprojects {
    repositories {
        google()
        mavenCentral()

        // 如果是CI构建，使用公司私有仓库
        if (project.isCiBuild) {
            maven {
                url 'https://maven.company.com/repo'
                credentials {
                    username = System.getenv('MAVEN_USER')
                    password = System.getenv('MAVEN_PASSWORD')
                }
            }
        }
    }
}

// 版本号自动递增任务
task incrementVersionCode {
    doLast {
        def currentCode = project.versionCode
        def nextCode = currentCode + 1

        ant.propertyfile(file: 'gradle.properties') {
            entry(key: 'VERSION_CODE', value: nextCode)
        }

        println "Version code incremented from ${currentCode} to ${nextCode}"
    }
}

// 生成版本信息任务
task generateVersionInfo {
    doLast {
        def versionInfoFile = file('src/main/assets/version_info.json')
        versionInfoFile.parentFile.mkdirs()

        def versionInfo = [
            versionName: project.versionName,
            versionCode: project.versionCode,
            buildTime: project.buildTime,
            buildTimestamp: project.buildTimestamp,
            gitCommit: project.gitCommit,
            gitBranch: project.gitBranch,
            isReleaseBuild: project.isReleaseBuild
        ]

        versionInfoFile.text = groovy.json.JsonBuilder(versionInfo).toPrettyString()
        println "Version info generated: ${versionInfoFile.absolutePath}"
    }
}

// 确保版本信息在构建前生成
preBuild.dependsOn generateVersionInfo
```

## 9.2 构建配置优化

### 9.2.1 性能优化配置

```groovy
// 构建性能优化
android {
    // 启用并行构建
    compileOptions {
        // Java编译器选项
        incremental = true
    }

    // 增量构建优化
    dexOptions {
        incremental = true
        javaMaxHeapSize "2g"
        maxProcessCount = Math.min(Runtime.runtime.availableProcessors(), 8)
    }

    // APT编译优化
    kapt {
        useBuildCache = true
        correctErrorTypes = true
    }
}

// 全局性能优化
gradle.projectsEvaluated {
    // 并行构建
    gradle.taskGraph.whenReady { graph ->
        if (project.hasProperty('parallel') || project.isCiBuild) {
            println "Enabling parallel build"
            gradle.startParameter.parallelProjectExecutionEnabled = true
        }
    }

    // 配置缓存
    if (project.hasProperty('configurationCache') || project.isCiBuild) {
        println "Enabling configuration cache"
        gradle.startParameter.configurationCache = true
    }
}

// 自定义构建缓存
buildCache {
    local {
        enabled = true
        directory = new File(rootDir, 'build-cache')
    }

    remote {
        enabled = project.hasProperty('buildCacheEnabled')
        url = project.findProperty('buildCacheUrl')
        push = project.hasProperty('buildCachePush')
    }
}

// 依赖解析优化
configurations.all {
    resolutionStrategy {
        cacheDynamicVersionsFor 10, 'minutes'
        cacheChangingModulesFor 0, 'seconds'

        preferProjectModules()
    }
}

// 任务优化
android.applicationVariants.all { variant ->
    // 优化合并资源任务
    variant.mergeResources.doLast {
        // 自定义资源优化逻辑
        println "Optimizing resources for ${variant.name}"
    }

    // 优化Dex任务
    variant.dexCompileProvider.configure {
        maxHeapSize = "2g"
        javaMaxHeapSize = "2g"
    }
}

// 清理任务优化
task cleanOptimized(type: Delete) {
    delete rootProject.buildDir

    // 也清理其他临时文件
    delete 'build-cache'
    delete '.gradle'
}

// 依赖分析任务
task analyzeDependencies {
    doLast {
        def configurations = project.configurations

        configurations.each { config ->
            if (config.canBeResolved) {
                println "=== ${config.name} ==="
                config.resolvedConfiguration.resolvedArtifacts.each { artifact ->
                    def file = artifact.file
                    def size = file.length()
                    println "${artifact.moduleVersion.id} - ${size} bytes"
                }
                println()
            }
        }
    }
}
```

### 9.2.2 多环境配置

```groovy
// 多环境配置管理
ext {
    // 环境定义
    environments = [
        'dev': [
            'apiBaseUrl': 'https://dev-api.example.com',
            'enableLogging': true,
            'enableCrashlytics': false,
            'enableAnalytics': false,
            'flavorDimension': 'env'
        ],
        'staging': [
            'apiBaseUrl': 'https://staging-api.example.com',
            'enableLogging': true,
            'enableCrashlytics': true,
            'enableAnalytics': false,
            'flavorDimension': 'env'
        ],
        'prod': [
            'apiBaseUrl': 'https://api.example.com',
            'enableLogging': false,
            'enableCrashlytics': true,
            'enableAnalytics': true,
            'flavorDimension': 'env'
        ]
    ]

    // 当前环境
    currentEnv = project.hasProperty('env') ? project.env : 'dev'
    envConfig = environments[currentEnv]
}

android {
    flavorDimensions envConfig.flavorDimension

    productFlavors {
        dev {
            dimension envConfig.flavorDimension
            applicationIdSuffix ".dev"
            versionNameSuffix "-dev"

            buildConfigField "String", "API_BASE_URL", "\"${envConfig.apiBaseUrl}\""
            buildConfigField "boolean", "ENABLE_LOGGING", "${envConfig.enableLogging}"
            buildConfigField "boolean", "ENABLE_CRASHLYTICS", "${envConfig.enableCrashlytics}"
            buildConfigField "boolean", "ENABLE_ANALYTICS", "${envConfig.enableAnalytics}"
        }

        staging {
            dimension envConfig.flavorDimension
            applicationIdSuffix ".staging"
            versionNameSuffix "-staging"

            buildConfigField "String", "API_BASE_URL", "\"${envConfig.apiBaseUrl}\""
            buildConfigField "boolean", "ENABLE_LOGGING", "${envConfig.enableLogging}"
            buildConfigField "boolean", "ENABLE_CRASHLYTICS", "${envConfig.enableCrashlytics}"
            buildConfigField "boolean", "ENABLE_ANALYTICS", "${envConfig.enableAnalytics}"
        }

        prod {
            dimension envConfig.flavorDimension

            buildConfigField "String", "API_BASE_URL", "\"${envConfig.apiBaseUrl}\""
            buildConfigField "boolean", "ENABLE_LOGGING", "${envConfig.enableLogging}"
            buildConfigField "boolean", "ENABLE_CRASHLYTICS", "${envConfig.enableCrashlytics}"
            buildConfigField "boolean", "ENABLE_ANALYTICS", "${envConfig.enableAnalytics}"
        }
    }

    // 根据环境应用不同的ProGuard规则
    buildTypes {
        release {
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'

            if (currentEnv == 'prod') {
                proguardFile file('proguard/proguard-prod.pro')
            }
        }
    }
}

// 环境特定依赖
dependencies {
    // 根据环境添加不同的依赖
    if (envConfig.enableLogging) {
        implementation 'com.jakewharton.timber:timber:5.0.1'
    }

    if (envConfig.enableCrashlytics) {
        implementation 'com.google.firebase:firebase-crashlytics:18.3.2'
    }

    if (envConfig.enableAnalytics) {
        implementation 'com.google.firebase:firebase-analytics:21.2.0'
    }
}

// 环境切换任务
task switchEnvironment {
    doLast {
        if (!project.hasProperty('targetEnv')) {
            throw new GradleException("Please specify target environment using -PtargetEnv=dev|staging|prod")
        }

        def targetEnv = project.property('targetEnv')
        if (!environments.containsKey(targetEnv)) {
            throw new GradleException("Unknown environment: ${targetEnv}")
        }

        // 修改gradle.properties文件
        def propsFile = file('gradle.properties')
        def properties = new Properties()

        if (propsFile.exists()) {
            propsFile.withInputStream { stream ->
                properties.load(stream)
            }
        }

        properties.env = targetEnv

        propsFile.withOutputStream { stream ->
            properties.store(stream, "Switched to ${targetEnv} environment")
        }

        println "Switched to ${targetEnv} environment"
    }
}

// 环境信息打印任务
task printEnvironmentInfo {
    doLast {
        println "=== Current Environment Information ==="
        println "Environment: ${currentEnv}"
        println "API Base URL: ${envConfig.apiBaseUrl}"
        println "Logging Enabled: ${envConfig.enableLogging}"
        println "Crashlytics Enabled: ${envConfig.enableCrashlytics}"
        println "Analytics Enabled: ${envConfig.enableAnalytics}"
        println "Application ID: ${android.defaultConfig.applicationId}"
        println "Version: ${android.defaultConfig.versionName} (${android.defaultConfig.versionCode})"
    }
}
```

## 9.3 自定义插件开发

### 9.3.1 Android构建插件

```groovy
// 自定义Android构建插件
package com.example.gradle

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.artifacts.Configuration
import org.gradle.api.file.FileCollection
import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.Exec
import com.android.build.gradle.AppPlugin
import com.android.build.gradle.LibraryPlugin

class AndroidBuildPlugin implements Plugin<Project> {
    void apply(Project project) {
        // 确保应用了Android插件
        if (!project.plugins.hasPlugin(AppPlugin) && !project.plugins.hasPlugin(LibraryPlugin)) {
            throw new IllegalStateException('Android plugin must be applied first')
        }

        // 创建扩展
        def extension = project.extensions.create('androidBuildConfig', AndroidBuildExtension)

        // 应用配置
        applyAndroidConfig(project, extension)

        // 添加任务
        addCustomTasks(project, extension)

        // 配置依赖
        configureDependencies(project, extension)
    }

    private void applyAndroidConfig(Project project, AndroidBuildExtension extension) {
        project.android {
            compileSdkVersion extension.compileSdkVersion
            buildToolsVersion extension.buildToolsVersion

            defaultConfig {
                minSdkVersion extension.minSdkVersion
                targetSdkVersion extension.targetSdkVersion
                versionCode extension.versionCode
                versionName extension.versionName

                // 默认配置
                buildConfigField "String", "BUILD_TYPE", "\"${project.gradle.startParameter.taskNames.join(',')}\""
                buildConfigField "long", "BUILD_TIME", "${System.currentTimeMillis()}L"

                // 应用签名配置
                if (extension.signingConfig) {
                    signingConfigs {
                        release {
                            storeFile file(extension.signingConfig.storeFile)
                            storePassword extension.signingConfig.storePassword
                            keyAlias extension.signingConfig.keyAlias
                            keyPassword extension.signingConfig.keyPassword
                        }
                    }
                }
            }

            buildTypes {
                debug {
                    applicationIdSuffix extension.debugSuffix
                    versionNameSuffix extension.debugVersionSuffix
                    debuggable true
                    minifyEnabled false
                }

                release {
                    debuggable false
                    minifyEnabled true

                    if (extension.signingConfig) {
                        signingConfig signingConfigs.release
                    }

                    proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'),
                                  'proguard-rules.pro'
                }
            }

            compileOptions {
                sourceCompatibility extension.javaVersion
                targetCompatibility extension.javaVersion
            }

            lintOptions {
                abortOnError extension.lintAbortOnError
                checkReleaseBuilds extension.lintCheckReleaseBuilds
            }
        }
    }

    private void addCustomTasks(Project project, AndroidBuildExtension extension) {
        // 版本信息生成任务
        project.task('generateAppVersion') {
            doLast {
                def versionFile = project.file("${project.android.sourceSets.main.assets.dir}/version.json")
                versionFile.parentFile.mkdirs()

                def versionInfo = [
                    versionName: project.android.defaultConfig.versionName,
                    versionCode: project.android.defaultConfig.versionCode,
                    buildTime: new Date().format('yyyy-MM-dd HH:mm:ss'),
                    buildTimestamp: System.currentTimeMillis(),
                    gitCommit: getGitCommit(),
                    gitBranch: getGitBranch()
                ]

                versionFile.text = groovy.json.JsonBuilder(versionInfo).toPrettyString()
                println "Generated app version: ${versionFile.absolutePath}"
            }
        }

        // 应用信息任务
        project.task('printAppInfo') {
            doLast {
                println "=== Application Information ==="
                println "Application ID: ${project.android.defaultConfig.applicationId}"
                println "Version: ${project.android.defaultConfig.versionName} (${project.android.defaultConfig.versionCode})"
                println "Min SDK: ${project.android.defaultConfig.minSdkVersion}"
                println "Target SDK: ${project.android.defaultConfig.targetSdkVersion}"
                println "Compile SDK: ${project.android.compileSdkVersion}"
                println "Build Tools: ${project.android.buildToolsVersion}"
            }
        }

        // APK签名信息任务
        project.task('printSigningInfo') {
            doLast {
                project.android.applicationVariants.all { variant ->
                    if (variant.buildType.name == 'release') {
                        println "=== ${variant.name} Signing Information ==="
                        println "Sign APK: ${variant.buildType.signingConfig != null}"
                        if (variant.buildType.signingConfig) {
                            println "Store File: ${variant.buildType.signingConfig.storeFile}"
                            println "Key Alias: ${variant.buildType.signingConfig.keyAlias}"
                        }
                    }
                }
            }
        }

        // APK分析任务
        project.task('analyzeApk') {
            doLast {
                project.android.applicationVariants.all { variant ->
                    if (variant.buildType.name == 'release') {
                        def apkFile = variant.outputs.first().outputFile
                        println "=== ${apkFile.name} Analysis ==="
                        println "Size: ${apkFile.length()} bytes"

                        if (apkFile.exists()) {
                            // 使用aapt获取详细信息
                            project.exec {
                                commandLine 'aapt', 'dump', 'badging', apkFile.absolutePath
                            }
                        }
                    }
                }
            }
        }

        // 确保版本信息在构建前生成
        project.tasks.matching { it.name == 'preBuild' }.all {
            it.dependsOn project.generateAppVersion
        }
    }

    private void configureDependencies(Project project, AndroidBuildExtension extension) {
        // 添加基础依赖
        project.dependencies {
            implementation 'androidx.core:core-ktx:1.9.0'
            implementation 'androidx.appcompat:appcompat:1.5.4'
            implementation 'com.google.android.material:material:1.6.1'
            implementation 'androidx.constraintlayout:constraintlayout:2.1.4'

            testImplementation 'junit:junit:4.13.2'
            androidTestImplementation 'androidx.test.ext:junit:1.1.3'
            androidTestImplementation 'androidx.test.espresso:espresso-core:3.4.0'
        }

        // 根据配置添加额外依赖
        extension.additionalLibraries.each { lib ->
            project.dependencies.add(lib.configuration, lib.dependency)
        }
    }

    private String getGitCommit() {
        try {
            return 'git rev-parse --short HEAD'.execute().text.trim()
        } catch (Exception e) {
            return 'unknown'
        }
    }

    private String getGitBranch() {
        try {
            return 'git rev-parse --abbrev-ref HEAD'.execute().text.trim()
        } catch (Exception e) {
            return 'unknown'
        }
    }
}

// 插件扩展类
class AndroidBuildExtension {
    int compileSdkVersion = 33
    String buildToolsVersion = "33.0.0"
    int minSdkVersion = 21
    int targetSdkVersion = 33
    int versionCode = 1
    String versionName = "1.0.0"
    String debugSuffix = ".debug"
    String debugVersionSuffix = "-debug"
    boolean lintAbortOnError = false
    boolean lintCheckReleaseBuilds = false
    SigningConfigExtension signingConfig
    List<LibraryDependency> additionalLibraries = []

    void signingConfig(Closure closure) {
        signingConfig = new SigningConfigExtension()
        closure.delegate = signingConfig
        closure.resolveStrategy = Closure.DELEGATE_FIRST
        closure()
    }

    void library(String configuration, String dependency) {
        additionalLibraries.add(new LibraryDependency(configuration, dependency))
    }
}

class SigningConfigExtension {
    String storeFile
    String storePassword
    String keyAlias
    String keyPassword

    void storeFile(String path) {
        this.storeFile = path
    }

    void storePassword(String password) {
        this.storePassword = password
    }

    void keyAlias(String alias) {
        this.keyAlias = alias
    }

    void keyPassword(String password) {
        this.keyPassword = password
    }
}

class LibraryDependency {
    String configuration
    String dependency

    LibraryDependency(String configuration, String dependency) {
        this.configuration = configuration
        this.dependency = dependency
    }
}

// 应用自定义插件的示例
// build.gradle
apply plugin: com.example.gradle.AndroidBuildPlugin

androidBuildConfig {
    compileSdkVersion 33
    minSdkVersion 21
    targetSdkVersion 33
    versionCode 2
    versionName "2.0.0"

    signingConfig {
        storeFile 'keystore/release.keystore'
        storePassword System.getenv('KEYSTORE_PASSWORD')
        keyAlias System.getenv('KEY_ALIAS')
        keyPassword System.getenv('KEY_PASSWORD')
    }

    library 'implementation', 'com.squareup.retrofit2:retrofit:2.9.0'
    library 'implementation', 'com.squareup.retrofit2:converter-gson:2.9.0'
    library 'testImplementation', 'org.mockito:mockito-core:4.6.1'
}
```

### 9.3.2 资源和代码生成插件

```groovy
// 资源生成插件
package com.example.gradle

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.Sync

class ResourceGeneratorPlugin implements Plugin<Project> {
    void apply(Project project) {
        // 创建扩展
        def extension = project.extensions.create('resourceGen', ResourceGeneratorExtension)

        // 添加资源生成任务
        project.android.sourceSets.main.res.srcDirs += project.file("${project.buildDir}/generated/res")

        project.task('generateResources', type: Sync) {
            def outputDir = project.file("${project.buildDir}/generated/res/values")

            from project.file('templates/res')
            into outputDir

            // 过滤和替换变量
            filter { line ->
                line.replace('${app_name}', extension.appName)
                      .replace('${app_version}', extension.appVersion)
                      .replace('${build_time}', new Date().format('yyyy-MM-dd HH:mm:ss'))
            }

            doLast {
                println "Generated resources in ${outputDir}"
            }
        }

        // 生成图标
        project.task('generateIcons') {
            def iconDir = project.file("${project.buildDir}/generated/res/mipmap-hdpi")
            iconDir.mkdirs()

            doLast {
                // 这里可以集成图标生成工具
                println "Generated app icons"
            }
        }

        // 确保资源生成在编译前完成
        project.tasks.whenTaskAdded { task ->
            if (task.name == 'mergeDebugResources' || task.name == 'mergeReleaseResources') {
                task.dependsOn project.generateResources
                task.dependsOn project.generateIcons
            }
        }
    }
}

class ResourceGeneratorExtension {
    String appName = "My App"
    String appVersion = "1.0.0"
    List<String> supportedLanguages = ["en", "zh", "ja"]
    boolean generateAdaptiveIcons = true
    boolean generateShortcuts = false

    void appName(String name) {
        this.appName = name
    }

    void appVersion(String version) {
        this.appVersion = version
    }

    void languages(String... languages) {
        this.supportedLanguages.clear()
        this.supportedLanguages.addAll(languages)
    }

    void adaptiveIcons(boolean enabled) {
        this.generateAdaptiveIcons = enabled
    }

    void shortcuts(boolean enabled) {
        this.generateShortcuts = enabled
    }
}

// 代码生成插件
package com.example.gradle

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.JavaExec
import org.gradle.api.file.FileCollection

class CodeGeneratorPlugin implements Plugin<Project> {
    void apply(Project project) {
        def extension = project.extensions.create('codeGen', CodeGeneratorExtension)

        // 添加生成代码的源目录
        project.android.sourceSets.main.java.srcDirs += project.file("${project.buildDir}/generated/java")

        // Model生成任务
        project.task('generateModels', type: JavaExec) {
            def outputDir = project.file("${project.buildDir}/generated/java/com/example/models")
            outputDir.mkdirs()

            classpath = project.files(project.projectDir) + project configurations.compileClasspath
            main = 'com.example.generator.ModelGenerator'

            args = [
                '--input', extension.modelSpecFile,
                '--output', outputDir.absolutePath,
                '--package', extension.modelPackage
            ]

            doFirst {
                println "Generating models from ${extension.modelSpecFile}"
            }

            doLast {
                println "Models generated in ${outputDir}"
            }
        }

        // API接口生成任务
        project.task('generateApiInterfaces', type: JavaExec) {
            def outputDir = project.file("${project.buildDir}/generated/java/com/example/api")
            outputDir.mkdirs()

            classpath = project.files(project.projectDir) + project configurations.compileClasspath
            main = 'com.example.generator.ApiGenerator'

            args = [
                '--input', extension.apiSpecFile,
                '--output', outputDir.absolutePath,
                '--package', extension.apiPackage
            ]

            doFirst {
                println "Generating API interfaces from ${extension.apiSpecFile}"
            }

            doLast {
                println "API interfaces generated in ${outputDir}"
            }
        }

        // 确保代码生成在编译前完成
        project.tasks.whenTaskAdded { task ->
            if (task.name.startsWith('compile') && task.name.endsWith('Java')) {
                task.dependsOn project.generateModels
                task.dependsOn project.generateApiInterfaces
            }
        }
    }
}

class CodeGeneratorExtension {
    String modelSpecFile = 'models.json'
    String apiSpecFile = 'api.json'
    String modelPackage = 'com.example.models'
    String apiPackage = 'com.example.api'
    boolean generateRetrofitInterfaces = true
    boolean generateRoomEntities = false

    void modelSpec(String file) {
        this.modelSpecFile = file
    }

    void apiSpec(String file) {
        this.apiSpecFile = file
    }

    void modelPackage(String pkg) {
        this.modelPackage = pkg
    }

    void apiPackage(String pkg) {
        this.apiPackage = pkg
    }

    void enableRetrofit(boolean enabled) {
        this.generateRetrofitInterfaces = enabled
    }

    void enableRoom(boolean enabled) {
        this.generateRoomEntities = enabled
    }
}
```

## 9.4 CI/CD集成最佳实践

### 9.4.1 Jenkins集成

```groovy
// Jenkinsfile.groovy (用于Jenkins Pipeline)
pipeline {
    agent any

    environment {
        ANDROID_HOME = '/opt/android-sdk'
        JAVA_HOME = '/opt/java'
        GRADLE_HOME = '/opt/gradle'
        KEYSTORE_PASSWORD = credentials('keystore-password')
        KEY_ALIAS = credentials('key-alias')
        KEY_PASSWORD = credentials('key-password')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm

                // 获取Git信息
                script {
                    env.GIT_COMMIT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.GIT_BRANCH = sh(script: 'git rev-parse --abbrev-ref HEAD', returnStdout: true).trim()
                    env.GIT_TAG = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
                }

                echo "Building commit ${env.GIT_COMMIT} on branch ${env.GIT_BRANCH}"
            }
        }

        stage('Setup') {
            steps {
                // 设置权限
                sh 'chmod +x gradlew'

                // 下载依赖
                sh './gradlew --no-daemon dependencies'

                // 检查环境
                sh 'echo "JAVA_HOME: $JAVA_HOME"'
                sh 'echo "ANDROID_HOME: $ANDROID_HOME"'
                sh './gradlew --version'
            }
        }

        stage('Lint') {
            steps {
                sh './gradlew --no-daemon lint'
                archiveArtifacts artifacts: 'build/reports/lint-results*.html', fingerprint: true
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'build/reports/lint-results',
                    reportFiles: '*.html',
                    reportName: 'Android Lint Report'
                ])
            }
        }

        stage('Test') {
            steps {
                sh './gradlew --no-daemon test'

                // 发布测试报告
                junit 'build/test-results/test/TEST-*.xml'

                // 发布测试覆盖率
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'build/reports/tests/test',
                    reportFiles: 'index.html',
                    reportName: 'Unit Test Report'
                ])
            }

            post {
                always {
                    // 发布JaCoCo覆盖率报告
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'build/reports/jacoco/test/html',
                        reportFiles: 'index.html',
                        reportName: 'Code Coverage Report'
                    ])
                }
            }
        }

        stage('Build') {
            steps {
                // 根据分支选择构建类型
                script {
                    if (env.BRANCH_NAME == 'main' || env.BRANCH_NAME == 'master') {
                        sh './gradlew --no-daemon assembleRelease'
                        archiveArtifacts artifacts: 'build/outputs/apk/release/*.apk', fingerprint: true
                    } else {
                        sh './gradlew --no-daemon assembleDebug'
                        archiveArtifacts artifacts: 'build/outputs/apk/debug/*.apk', fingerprint: true
                    }
                }
            }
        }

        stage('UI Tests') {
            when {
                expression { env.BRANCH_NAME == 'main' || env.BRANCH_NAME == 'master' }
            }
            steps {
                // 配置Android模拟器
                sh '''
                    echo "no" | ${ANDROID_HOME}/tools/bin/sdkmanager --install "system-images;android-30;google_apis;x86_64"
                    echo "no" | ${ANDROID_HOME}/tools/bin/avdmanager create avd -n test -k "system-images;android-30;google_apis;x86_64"
                    ${ANDROID_HOME}/tools/emulator -avd test -no-window -no-audio &
                    sleep 60
                '''

                // 运行UI测试
                sh './gradlew --no-daemon connectedAndroidTest'

                // 发布UI测试报告
                archiveArtifacts artifacts: 'build/reports/androidTests/connected/**/*.html', fingerprint: true
            }
        }

        stage('Deploy') {
            when {
                anyOf {
                    branch 'main'
                    branch 'master'
                    tag pattern: "v\\d+\\.\\d+\\.\\d+", comparator: "REGEXP"
                }
            }
            steps {
                script {
                    if (env.TAG_NAME) {
                        // 发布到Google Play Store
                        sh './gradlew --no-daemon publishReleaseApk'

                        // 创建GitHub Release
                        sh '''
                            gh release create ${TAG_NAME} \
                                build/outputs/apk/release/*.apk \
                                --title "Release ${TAG_NAME}" \
                                --notes "Automated release ${TAG_NAME}"
                        '''
                    } else {
                        // 部署到测试环境
                        sh './gradlew --no-daemon uploadToTest'

                        // 通知测试团队
                        slackSend(
                            channel: '#android-testing',
                            color: 'good',
                            message: "New test build deployed: ${BUILD_URL}artifact/"
                        )
                    }
                }
            }
        }
    }

    post {
        always {
            // 清理工作空间
            cleanWs()
        }

        failure {
            // 发送失败通知
            mail to: 'dev-team@example.com',
                subject: "Build Failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}"
        }

        success {
            // 发送成功通知（仅主分支）
            script {
                if (env.BRANCH_NAME == 'main' || env.BRANCH_NAME == 'master') {
                    slackSend(
                        channel: '#android-builds',
                        color: 'good',
                        message: "Build succeeded: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
                    )
                }
            }
        }
    }
}
```

### 9.4.2 GitHub Actions集成

```groovy
// .github/workflows/android.yml
name: Android CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

env:
  JAVA_VERSION: '11'
  ANDROID_SDK_VERSION: '33'
  GRADLE_OPTS: '-Dorg.gradle.daemon=false -Dorg.gradle.workers.max=2'

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'

    - name: Set up Android SDK
      uses: android-actions/setup-android@v2

    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
        restore-keys: |
          ${{ runner.os }}-gradle-

    - name: Cache Android build cache
      uses: actions/cache@v3
      with:
        path: |
          ~/.android/build-cache
        key: ${{ runner.os }}-android-build-cache-${{ hashFiles('**/*.gradle*') }}
        restore-keys: |
          ${{ runner.os }}-android-build-cache-

    - name: Grant execute permission for gradlew
      run: chmod +x gradlew

    - name: Run unit tests
      run: ./gradlew test

    - name: Run Lint
      run: ./gradlew lint

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: build/reports/tests/

    - name: Upload lint results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: lint-results
        path: build/reports/lint-results/

  build:
    needs: test
    runs-on: ubuntu-latest

    strategy:
      matrix:
        variant: [debug, release]

    steps:
    - uses: actions/checkout@v3

    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'

    - name: Set up Android SDK
      uses: android-actions/setup-android@v2

    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}

    - name: Grant execute permission for gradlew
      run: chmod +x gradlew

    - name: Build APK
      run: ./gradlew assemble${{ matrix.variant }}

    - name: Upload APK
      uses: actions/upload-artifact@v3
      with:
        name: app-${{ matrix.variant }}
        path: build/outputs/apk/${{ matrix.variant }}/*.apk

  integration-test:
    needs: build
    runs-on: macos-latest  # macOS对Android模拟器支持更好

    steps:
    - uses: actions/checkout@v3

    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'

    - name: Set up Android SDK
      uses: android-actions/setup-android@v2

    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}

    - name: Cache Android build cache
      uses: actions/cache@v3
      with:
        path: |
          ~/.android/build-cache
        key: ${{ runner.os }}-android-build-cache-${{ hashFiles('**/*.gradle*') }}

    - name: Grant execute permission for gradlew
      run: chmod +x gradlew

    - name: Build debug APK
      run: ./gradlew assembleDebug

    - name: Run instrumented tests
      uses: reactivecircus/android-emulator-runner@v2
      with:
        api-level: 29
        target: default
        arch: x86
        script: ./gradlew connectedDebugAndroidTest

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: build/reports/androidTests/connected/

  deploy:
    needs: [build, integration-test]
    runs-on: ubuntu-latest
    if: github.event_name == 'release'

    steps:
    - uses: actions/checkout@v3

    - name: Set up JDK 11
      uses: actions/setup-java@v3
      with:
        java-version: '11'
        distribution: 'temurin'

    - name: Set up Android SDK
      uses: android-actions/setup-android@v2

    - name: Cache Gradle packages
      uses: actions/cache@v3
      with:
        path: |
          ~/.gradle/caches
          ~/.gradle/wrapper
        key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}

    - name: Grant execute permission for gradlew
      run: chmod +x gradlew

    - name: Decode Keystore
      env:
        ENCODED_STRING: ${{ secrets.KEYSTORE_BASE64 }}
      run: |
        echo $ENCODED_STRING | base64 -di > keystore/release.keystore

    - name: Build Signed Release APK
      env:
        KEYSTORE_PASSWORD: ${{ secrets.KEYSTORE_PASSWORD }}
        KEY_ALIAS: ${{ secrets.KEY_ALIAS }}
        KEY_PASSWORD: ${{ secrets.KEY_PASSWORD }}
      run: ./gradlew assembleRelease

    - name: Sign APK
      run: |
        jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
          -keystore keystore/release.keystore \
          -storepass $KEYSTORE_PASSWORD \
          -keypass $KEY_PASSWORD \
          build/outputs/apk/release/*.apk $KEY_ALIAS

    - name: Zip Align APK
      run: |
        ${ANDROID_HOME}/build-tools/${{ env.ANDROID_SDK_VERSION }}/zipalign \
          -v 4 build/outputs/apk/release/*.apk \
          build/outputs/apk/release/app-release-aligned.apk

    - name: Upload Release Asset
      uses: actions/upload-release-asset@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        upload_url: ${{ github.event.release.upload_url }}
        asset_path: build/outputs/apk/release/app-release-aligned.apk
        asset_name: app-release.apk
        asset_content_type: application/vnd.android.package-archive

    - name: Deploy to Google Play Store
      uses: r0adkll/upload-google-play@v1
      with:
        serviceAccountJsonPlainText: ${{ secrets.GOOGLE_PLAY_SERVICE_ACCOUNT }}
        packageName: com.example.myapp
        releaseFiles: build/outputs/apk/release/app-release-aligned.apk
        track: production
```

## 本章小结

Groovy在Android开发中发挥着重要作用，特别是在构建系统、配置管理和自动化流程方面。

### 核心概念回顾

1. **Gradle在Android中的基础应用**：构建配置、依赖管理、任务定义
2. **动态版本和配置管理**：环境配置、版本自动化
3. **构建配置优化**：性能优化、多环境支持
4. **自定义插件开发**：构建器模式、代码生成
5. **CI/CD集成最佳实践**：Jenkins、GitHub Actions集成

### 实战应用

✅ **掌握Android Gradle插件**：构建配置、变体管理
✅ **学会多环境配置**：开发、测试、生产环境
✅ **掌握性能优化**：构建速度、缓存策略
✅ **开发自定义插件**：资源生成、代码生成
✅ **集成CI/CD**：自动化构建、测试、部署

### 最佳实践

- **模块化构建配置**：将复杂的构建逻辑分解到插件中
- **环境隔离**：使用不同的构建类型和产品风味
- **性能优化**：合理使用缓存、并行构建
- **自动化测试**：集成单元测试和UI测试
- **持续集成**：建立完整的CI/CD流水线

---

**第二部分学习完成！现在你已经掌握了Groovy的进阶特性，包括MOP、AST转换、Gradle构建和Android应用。这些知识为学习DSL开发奠定了坚实的基础。**

**总计已完成约50,000字的教程内容，涵盖了从Groovy基础到高级应用的完整学习路径！**