# 第三章：Gradle核心概念深度解析

## 3.1 Project（项目）概念

### 3.1.1 Project的定义与生命周期

**Project概念定义：**

在Gradle中，Project是构建的基本单元。每个Gradle构建至少包含一个Project（根Project），在多模块构建中可能包含多个子Project。与Maven的Module概念相比，Gradle的Project更加灵活和强大。

**Project与Maven Module对比：**

| 特性 | Maven Module | Gradle Project |
|------|--------------|----------------|
| 定义方式 | 父POM中的modules声明 | settings.gradle中的include声明 |
| 配置继承 | 父POM自动继承 | 通过subprojects/allprojects配置 |
| 依赖关系 | 通过dependency声明 | 通过project(':module-name')声明 |
| 配置灵活性 | 受限于XML结构 | 支持编程式配置 |
| 任务定义 | 插件预定义 | 可动态定义任务 |

**Project生命周期：**

```groovy
// build.gradle - Project生命周期演示

// 1. 项目评估前钩子
beforeEvaluate {
    println "Project ${project.name} is about to be evaluated"
}

// 2. 项目评估后钩子
afterEvaluate {
    println "Project ${project.name} has been evaluated"
    println "Available tasks: ${tasks.names}"
}

// 3. 项目加载完成钩子
gradle.projectsLoaded {
    println "All projects have been loaded"
}

// 4. 设置阶段完成钩子
gradle.settingsEvaluated {
    println "Settings have been evaluated"
}

// 5. 任务图创建完成钩子
gradle.taskGraph.whenReady { graph ->
    println "Task graph is ready with ${graph.allTasks.size()} tasks"
}

// 6. 任务执行前钩子
gradle.taskGraph.beforeTask { task ->
    println "Executing task: ${task.path}"
}

// 7. 任务执行后钩子
gradle.taskGraph.afterTask { task, taskState ->
    if (taskState.failure) {
        println "Task ${task.path} failed: ${taskState.failure}"
    } else {
        println "Task ${task.path} completed successfully"
    }
}

// 8. 构建完成钩子
gradle.buildFinished { result ->
    println "Build finished with result: ${result}"
}
```

**Project属性和方法：**

```groovy
// build.gradle - Project核心属性和方法演示

// 1. 项目基本信息
println "Project name: ${name}"                    // 项目名称
println "Project path: ${path}"                    // 项目路径
println "Project description: ${description}"      // 项目描述
println "Project group: ${group}"                  // 项目组ID
println "Project version: ${version}"              // 项目版本

// 2. 项目目录信息
println "Project directory: ${projectDir}"         // 项目目录
println "Build directory: ${buildDir}"             // 构建目录
println "Root directory: ${rootDir}"               // 根项目目录

// 3. 项目状态信息
println "Project state: ${state}"                  // 项目状态
println "Gradle version: ${gradle.gradleVersion}"  // Gradle版本

// 4. 项目配置方法
// 4.1 扩展属性
ext {
    appName = 'My Application'
    appVersion = '1.0.0'
    buildTimestamp = new Date().format('yyyy-MM-dd HH:mm:ss')
}

// 4.2 配置块
configurations {
    // 自定义配置
    provided
    embed
}

// 4.3 依赖管理
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation 'org.junit.jupiter:junit-jupiter'
}

// 4.4 任务创建
task hello {
    doLast {
        println "Hello from project: ${project.name}"
        println "App name: ${appName}"
        println "Build time: ${buildTimestamp}"
    }
}

// 4.5 文件操作
def configFile = file('src/main/resources/application.yml')
if (configFile.exists()) {
    println "Config file exists: ${configFile.absolutePath}"
}

// 4.6 属性访问
project.ext.customProperty = 'Custom Value'
println "Custom property: ${project.ext.customProperty}"
```

### 3.1.2 Project层次结构

**单项目结构：**

```groovy
// 单项目结构示例
project-root/
├── build.gradle          # 唯一的构建脚本
├── settings.gradle       # 最简配置
└── src/                  # 源码目录

// settings.gradle
rootProject.name = 'single-project'

// build.gradle
println "This is a single project: ${project.name}"
println "Root project: ${rootProject.name}"
println "Is this root project: ${project == rootProject}"
```

**多项目结构：**

```groovy
// 多项目结构示例
multi-module-project/
├── build.gradle          # 根项目构建脚本
├── settings.gradle       # 项目配置
├── module-core/
│   └── build.gradle      # 核心模块构建脚本
├── module-web/
│   └── build.gradle      # Web模块构建脚本
└── module-api/
    └── build.gradle      # API模块构建脚本

// settings.gradle
rootProject.name = 'multi-module-project'

include 'module-core'
include 'module-web'
include 'module-api'

// 子项目重命名
project(':module-core').name = 'core'
project(':module-web').name = 'web'
project(':module-api').name = 'api'

// build.gradle (根项目)
println "Root project: ${name}"

// 子项目通用配置
subprojects { subproject ->
    println "Configuring subproject: ${subproject.name}"

    // 应用基础插件
    apply plugin: 'java'

    // 配置Java版本
    java {
        sourceCompatibility = JavaVersion.VERSION_17
    }

    // 仓库配置
    repositories {
        mavenCentral()
    }
}

// 特定项目配置
project(':core') {
    dependencies {
        implementation 'org.apache.commons:commons-lang3'
    }
}

project(':web') {
    dependencies {
        implementation project(':core')
        implementation 'org.springframework.boot:spring-boot-starter-web'
    }
}

project(':api') {
    dependencies {
        implementation project(':core')
    }
}
```

**项目层次结构操作：**

```groovy
// build.gradle - 项目层次结构操作示例

// 1. 遍历所有项目
allprojects { project ->
    println "Project: ${project.path} - ${project.name}"
}

// 2. 遍历子项目
subprojects { subproject ->
    println "Subproject: ${subproject.path} - ${subproject.name}"
}

// 3. 访问特定项目
def coreProject = project(':module-core')
if (coreProject) {
    println "Core project found: ${coreProject.name}"
}

// 4. 检查项目关系
println "Is root project: ${project == rootProject}"
println "Parent project: ${project.parent?.name ?: 'none'}"
println "Child projects: ${project.childProjects.keySet()}"

// 5. 项目评估
gradle.afterProject { project, state ->
    if (state.failure) {
        println "Project ${project.name} evaluation failed"
    } else {
        println "Project ${project.name} evaluated successfully"
    }
}
```

### 3.1.3 Project属性和方法

**Project核心属性：**

```groovy
// build.gradle - Project核心属性详解

// 1. 标识属性
println "=== Project Identification ==="
println "Name: ${name}"                    // 项目名称
println "Path: ${path}"                    // 项目路径
println "Display name: ${displayName}"     // 显示名称
println "Description: ${description}"      // 项目描述

// 2. 版本信息
println "\n=== Version Information ==="
println "Group: ${group}"                  // 组ID
println "Version: ${version}"              // 版本号
println "Status: ${status}"                // 项目状态

// 3. 目录属性
println "\n=== Directory Properties ==="
println "Project dir: ${projectDir}"       // 项目目录
println "Build dir: ${buildDir}"           // 构建目录
println "Root dir: ${rootDir}"             // 根目录
println "Gradle user home: ${gradle.gradleUserHomeDir}"

// 4. 项目状态
println "\n=== Project State ==="
println "State: ${state}"                  // 项目状态对象
println "Executed: ${state.executed}"      // 是否已执行
println "Failure: ${state.failure}"        // 失败信息
println "Reexecuted: ${state.reexecuted}"  // 是否重新执行

// 5. 构建相关信息
println "\n=== Build Information ==="
println "Gradle version: ${gradle.gradleVersion}"
println "Gradle home: ${gradle.gradleHomeDir}"
println "Start parameter: ${gradle.startParameter}"
```

**Project核心方法：**

```groovy
// build.gradle - Project核心方法演示

// 1. 文件操作方法
println "=== File Operations ==="

// file()方法 - 创建文件对象
def configFile = file('src/main/resources/application.yml')
println "Config file: ${configFile.absolutePath}"

// files()方法 - 创建文件集合
def resourceFiles = files('src/main/resources')
println "Resource files: ${resourceFiles.files}"

// fileTree()方法 - 创建文件树
def javaFiles = fileTree('src/main/java') {
    include '**/*.java'
    exclude '**/generated/**'
}
println "Java files count: ${javaFiles.files.size()}"

// mkdir()方法 - 创建目录
def outputDir = mkdir('build/custom-output')
println "Created directory: ${outputDir.absolutePath}"

// 2. 依赖操作方法
println "\n=== Dependency Operations ==="

// dependencies()方法 - 配置依赖
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation 'org.junit.jupiter:junit-jupiter'
}

// configurations()方法 - 配置依赖配置
configurations {
    implementation.canBeResolved = true
    compileClasspath.extendsFrom(implementation)
}

// 3. 任务操作方法
println "\n=== Task Operations ==="

// task()方法 - 创建任务
task customTask {
    doLast {
        println "Executing custom task"
    }
}

// tasks()方法 - 访问任务集合
tasks.register('anotherTask') {
    doLast {
        println "Executing another task"
    }
}

// 4. 属性操作方法
println "\n=== Property Operations ==="

// ext属性扩展
ext {
    customProperty = 'Custom Value'
    buildInfo = [
        timestamp: new Date().format('yyyy-MM-dd HH:mm:ss'),
        version: project.version,
        author: System.getProperty('user.name')
    ]
}

// hasProperty()方法 - 检查属性存在
println "Has customProperty: ${hasProperty('customProperty')}"
println "Has missingProperty: ${hasProperty('missingProperty')}"

// findProperty()方法 - 安全获取属性
println "Custom property value: ${findProperty('customProperty')}"
println "Missing property value: ${findProperty('missingProperty')}"

// 5. 项目操作方法
println "\n=== Project Operations ==="

// project()方法 - 访问项目
def coreProject = project(':module-core')
if (coreProject) {
    println "Core project version: ${coreProject.version}"
}

// evaluationDependsOn()方法 - 设置评估依赖
evaluationDependsOn(':module-core')

// 6. 日志方法
println "\n=== Logging Methods ==="

logger.lifecycle('Lifecycle log message')
logger.info('Info log message')
logger.debug('Debug log message')
logger.warn('Warning log message')
logger.error('Error log message')
```

### 3.1.4 多项目构建基础

**多项目构建最佳实践：**

```groovy
// build.gradle - 多项目构建最佳实践

// 1. 项目结构定义
// settings.gradle
rootProject.name = 'enterprise-application'

// 核心模块
include 'common:utils'
include 'common:domain'
include 'common:security'

// 基础设施模块
include 'infrastructure:persistence'
include 'infrastructure:messaging'
include 'infrastructure:config'

// 应用模块
include 'application:api'
include 'application:web'
include 'application:batch'
include 'application:admin'

// 测试模块
include 'test:integration'
include 'test:performance'

// 2. 根项目配置
// build.gradle (根项目)

// 插件管理
plugins {
    id 'java-platform' version '0.0.1'
    id 'io.spring.dependency-management' version '1.1.4' apply false
}

// 依赖版本管理
javaPlatform {
    allowDependencies()

    dependencies {
        api platform('org.springframework.boot:spring-boot-dependencies:3.2.0')

        // 工具库版本
        api 'org.apache.commons:commons-lang3:3.13.0'
        api 'com.google.guava:guava:32.1.3-jre'

        // 测试依赖版本
        api 'org.junit.jupiter:junit-jupiter:5.10.0'
        api 'org.mockito:mockito-core:5.6.0'
        api 'org.assertj:assertj-core:3.24.2'
    }
}

// 子项目通用配置
subprojects { subproject ->
    // 应用基础插件
    apply plugin: 'java-library'
    apply plugin: 'groovy'  // 支持Groovy测试

    // Java配置
    java {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17

        withSourcesJar()
        withJavadocJar()
    }

    // 仓库配置
    repositories {
        mavenCentral()
        maven { url 'https://repo.spring.io/milestone' }
    }

    // 依赖管理
    dependencyManagement {
        imports {
            mavenBom rootProject.dependencies.platform('org.springframework.boot:spring-boot-dependencies:3.2.0')
        }
    }

    // 通用依赖
    dependencies {
        // 编译时依赖
        compileOnly 'org.projectlombok:lombok'
        annotationProcessor 'org.projectlombok:lombok'

        // 测试依赖
        testImplementation 'org.junit.jupiter:junit-jupiter'
        testImplementation 'org.mockito:mockito-core'
        testImplementation 'org.assertj:assertj-core'
        testImplementation 'org.spockframework:spock-core:2.4-M1-groovy-4.0'

        // 运行时依赖
        runtimeOnly 'ch.qos.logback:logback-classic'
    }

    // 测试配置
    test {
        useJUnitPlatform()

        // JVM参数
        jvmArgs = [
            '-Dspring.profiles.active=test',
            '-Dfile.encoding=UTF-8',
            '-Djava.awt.headless=true'
        ]

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

// 3. 模块特定配置

// 通用模块配置
configure(subprojects.findAll { it.path.startsWith(':common') }) { commonModule ->
    group = 'com.example.enterprise.common'

    // 通用模块不生成可执行JAR
    jar {
        enabled = true
        archiveClassifier.set('')
    }

    dependencies {
        api 'org.apache.commons:commons-lang3'
        api 'org.slf4j:slf4j-api'
    }
}

// 基础设施模块配置
configure(subprojects.findAll { it.path.startsWith(':infrastructure') }) { infraModule ->
    group = 'com.example.enterprise.infrastructure'

    dependencies {
        implementation project(':common:domain')

        if (infraModule.name == 'persistence') {
            implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
            runtimeOnly 'mysql:mysql-connector-java'
            runtimeOnly 'com.h2database:h2'
        }

        if (infraModule.name == 'messaging') {
            implementation 'org.springframework.boot:spring-boot-starter-amqp'
        }
    }
}

// 应用模块配置
configure(subprojects.findAll { it.path.startsWith(':application') }) { appModule ->
    group = 'com.example.enterprise.application'

    // 应用模块应用Spring Boot插件
    apply plugin: 'org.springframework.boot'

    dependencies {
        implementation project(':common:domain')
        implementation project(':common:security')

        if (appModule.name == 'web') {
            implementation 'org.springframework.boot:spring-boot-starter-web'
            implementation 'org.springframework.boot:spring-boot-starter-actuator'
            developmentOnly 'org.springframework.boot:spring-boot-devtools'
        }

        if (appModule.name == 'batch') {
            implementation 'org.springframework.boot:spring-boot-starter-batch'
        }

        if (appModule.name == 'admin') {
            implementation 'org.springframework.boot:spring-boot-starter-web'
            implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
        }
    }
}

// 测试模块配置
configure(subprojects.findAll { it.path.startsWith(':test') }) { testModule ->
    group = 'com.example.enterprise.test'

    dependencies {
        implementation project(':application:web')
        implementation project(':infrastructure:persistence')

        if (testModule.name == 'integration') {
            implementation 'org.springframework.boot:spring-boot-starter-test'
            implementation 'org.testcontainers:junit-jupiter'
            implementation 'org.testcontainers:mysql'
        }

        if (testModule.name == 'performance') {
            implementation 'org.springframework.boot:spring-boot-starter-test'
            implementation 'org.springframework.boot:spring-boot-starter-actuator'
            implementation 'com.github.tomakehurst:wiremock-jre8:2.35.0'
        }
    }
}

// 4. 自定义任务

// 构建所有应用
task buildApplications {
    group = 'build'
    description = 'Build all application modules'

    dependsOn subprojects.findAll {
        it.path.startsWith(':application')
    }.collect { "${it.path}:build" }
}

// 构建所有模块
task buildAll {
    group = 'build'
    description = 'Build all modules'

    dependsOn subprojects.collect { "${it.path}:build" }
}

// 运行所有测试
task testAll {
    group = 'verification'
    description = 'Run all tests across all modules'

    dependsOn subprojects.collect { "${it.path}:test" }
}

// 集成测试
task runIntegrationTests {
    group = 'verification'
    description = 'Run integration tests'

    dependsOn ':test:integration:test'
}

// 性能测试
task runPerformanceTests {
    group = 'verification'
    description = 'Run performance tests'

    dependsOn ':test:performance:test'
}

// 生成项目报告
task projectReport {
    group = 'reporting'
    description = 'Generate project structure report'

    doLast {
        def reportFile = file("$buildDir/project-report.txt")
        reportFile.parentFile.mkdirs()

        reportFile.text = """
            Enterprise Application Project Report
            ======================================

            Project Name: ${rootProject.name}
            Total Modules: ${subprojects.size()}

            Module Structure:
            ================

            Common Modules:
            ${subprojects.findAll { it.path.startsWith(':common') }.collect { "  - ${it.name}" }.join('\n')}

            Infrastructure Modules:
            ${subprojects.findAll { it.path.startsWith(':infrastructure') }.collect { "  - ${it.name}" }.join('\n')}

            Application Modules:
            ${subprojects.findAll { it.path.startsWith(':application') }.collect { "  - ${it.name}" }.join('\n')}

            Test Modules:
            ${subprojects.findAll { it.path.startsWith(':test') }.collect { "  - ${it.name}" }.join('\n')}

            Dependencies Summary:
            ====================
            Total Dependencies: ${subprojects.collect { it.configurations.implementation.dependencies.size() }.sum()}

            Build Information:
            ==================
            Gradle Version: ${gradle.gradleVersion}
            Java Version: ${System.getProperty('java.version')}
            Build Time: ${new Date().format('yyyy-MM-dd HH:mm:ss')}
        """.stripIndent()

        println "Project report generated: ${reportFile.absolutePath}"
    }
}
```

## 3.2 Task（任务）系统详解

### 3.2.1 Task的定义与类型

**Task基础概念：**

Task是Gradle中的工作单元，每个Task执行一个特定的操作（如编译代码、运行测试、打包JAR等）。与Maven的Phase概念不同，Gradle的Task更加细粒度和灵活。

**Task类型体系：**

```groovy
// build.gradle - Task类型体系演示

// 1. DefaultTask - 基础任务类型
task basicTask(type: DefaultTask) {
    description 'A basic task example'
    group 'custom'

    doLast {
        println "This is a basic task"
    }
}

// 2. Copy - 文件复制任务
task copyConfigFiles(type: Copy) {
    description 'Copy configuration files'
    group 'build'

    from 'src/main/resources'
    into "$buildDir/config"
    include '**/*.yml', '**/*.properties'

    // 文件过滤
    filter { line ->
        line.replace('\${project.version}', project.version.toString())
    }
}

// 3. Delete - 文件删除任务
task cleanCustom(type: Delete) {
    description 'Clean custom directories'
    group 'build'

    delete 'build/tmp', 'build/logs'
}

// 4. Zip - 压缩任务
task createDist(type: Zip) {
    description 'Create distribution package'
    group 'distribution'

    from jar.outputs.files
    from 'src/main/resources'
    into 'lib'

    archiveFileName = "${project.name}-${project.version}.zip"
    destinationDirectory = file("$buildDir/distributions")
}

// 5. Tar - 打包任务
task createTarDist(type: Tar) {
    description 'Create TAR distribution'
    group 'distribution'

    from jar.outputs.files
    from 'src/main/resources'

    compression = Compression.GZIP
    archiveFileName = "${project.name}-${project.version}.tar.gz"
    destinationDirectory = file("$buildDir/distributions")
}

// 6. JavaCompile - Java编译任务
task customJavaCompile(type: JavaCompile) {
    description 'Custom Java compilation'
    group 'build'

    source = fileTree('src/custom/java')
    classpath = sourceSets.main.compileClasspath
    destinationDirectory = file("$buildDir/classes/custom")
}

// 7. Test - 测试任务
task customTest(type: Test) {
    description 'Custom test execution'
    group 'verification'

    testClassesDirs = sourceSets.test.output.classesDirs
    classpath = sourceSets.test.runtimeClasspath

    include '**/*Test.class'
    exclude '**/*IntegrationTest.class'

    // 测试配置
    testLogging {
        events "passed", "skipped", "failed"
        exceptionFormat "full"
    }
}

// 8. Jar - JAR打包任务
task customJar(type: Jar) {
    description 'Create custom JAR'
    group 'build'

    from sourceSets.main.output
    from configurations.runtimeClasspath

    manifest {
        attributes(
            'Implementation-Title': project.name,
            'Implementation-Version': project.version,
            'Main-Class': 'com.example.CustomApplication'
        )
    }

    archiveClassifier.set('custom')
}

// 9. Exec - 执行外部命令任务
task runCommand(type: Exec) {
    description 'Execute external command'
    group 'tools'

    commandLine 'echo', 'Hello from Exec task'

    // 条件执行
    onlyIf { project.hasProperty('runExternal') }
}

// 10. Sync - 同步任务
task syncFiles(type: Sync) {
    description 'Sync files between directories'
    group 'build'

    from 'src/main/resources'
    into "$buildDir/synced-resources"

    preserve {
        include 'application*.yml'
    }
}
```

**自定义Task类型：**

```groovy
// buildSrc/src/main/groovy/com/example/tasks/CustomTask.groovy
package com.example.tasks

import org.gradle.api.DefaultTask
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.TaskAction

class CustomTask extends DefaultTask {

    // 任务输入
    @Input
    String message = 'Default message'

    @InputFile
    File inputFile

    // 任务输出
    @OutputFile
    File outputFile

    // 构造函数
    CustomTask() {
        group = 'custom'
        description = 'A custom task example'
    }

    // 任务执行逻辑
    @TaskAction
    void execute() {
        println "Executing custom task with message: $message"

        if (inputFile && inputFile.exists()) {
            println "Processing input file: ${inputFile.name}"
            def content = inputFile.text
            def processedContent = "${message}\n${content}"

            if (outputFile) {
                outputFile.parentFile.mkdirs()
                outputFile.text = processedContent
                println "Output written to: ${outputFile.absolutePath}"
            }
        } else {
            println "No input file provided"
        }
    }
}

// build.gradle - 使用自定义Task
task processConfigFile(type: CustomTask) {
    message = 'Configuration processed'
    inputFile = file('src/main/resources/application.yml')
    outputFile = file("$buildDir/processed-config.yml")
}
```

### 3.2.2 Task生命周期（初始化、配置、执行）

**Task生命周期详解：**

```groovy
// build.gradle - Task生命周期演示

// 1. 任务创建阶段
task lifecycleDemo {
    description 'Demonstrate task lifecycle'
    group 'demo'

    // 2. 任务配置阶段 - 在配置阶段执行
    println "Configuration phase: Task lifecycleDemo is being configured"

    // 配置属性
    ext.configuredProperty = 'Configured in configuration phase'

    // 3. 任务执行阶段 - 在执行阶段执行
    doFirst {
        println "Execution phase - doFirst: Before main action"
    }

    // 主要执行逻辑
    doLast {
        println "Execution phase - doLast: Main action"
        println "Configured property: ${configuredProperty}"
    }

    // 多个doLast操作
    doLast {
        println "Execution phase - doLast: Additional action"
    }
}

// 4. 生命周期钩子演示
task lifecycleHooks {
    description 'Show lifecycle hooks'
    group 'demo'

    // 任务执行前钩子
    onlyIf {
        println "onlyIf check: Determining if task should execute"
        return !project.hasProperty('skipDemo')
    }

    // 依赖任务执行前
    dependsOn tasks.register('preHookTask') {
        doLast {
            println "Pre-hook task executed"
        }
    }

    // 最终化任务
    finalizedBy tasks.register('postHookTask') {
        doLast {
            println "Post-hook task executed"
        }
    }

    doLast {
        println "Main task executed"
    }
}

// 5. 动态任务创建演示
tasks.register('dynamicTask') {
    println "Dynamic task being configured"

    doLast {
        println "Dynamic task executed"
    }
}

// 6. 任务配置阶段演示
println "=== Build Script Configuration Phase ==="
println "All tasks being configured..."

// 7. 项目评估钩子
gradle.afterProject { project, state ->
    if (state.failure) {
        println "Project ${project.name} evaluation failed"
    } else {
        println "Project ${project.name} evaluation completed"
    }
}

// 8. 任务图准备钩子
gradle.taskGraph.whenReady { graph ->
    println "=== Task Graph Ready ==="
    println "Tasks to be executed:"
    graph.allTasks.each { task ->
        println "  - ${task.path} (${task.class.simpleName})"
    }
}

// 9. 任务执行前钩子
gradle.taskGraph.beforeTask { task ->
    println "Before executing: ${task.path}"
}

// 10. 任务执行后钩子
gradle.taskGraph.afterTask { task, taskState ->
    if (taskState.failure) {
        println "Task ${task.path} FAILED: ${taskState.failure}"
    } else if (taskState.skipped) {
        println "Task ${task.path} SKIPPED: ${taskState.skipMessage}"
    } else if (taskState.upToDate) {
        println "Task ${task.path} UP-TO-DATE"
    } else {
        println "Task ${task.path} COMPLETED"
    }
}
```

**Task配置vs执行阶段对比：**

```groovy
// build.gradle - 配置阶段vs执行阶段对比

// 错误示例 - 在配置阶段执行耗时操作
task badExample {
    println "Configuration phase: Starting expensive operation"

    // 这会在配置阶段执行，每次构建都会执行
    def result = expensiveOperation()

    doLast {
        println "Execution phase: Using result from configuration phase"
        println "Result: $result"
    }
}

// 正确示例 - 将耗时操作延迟到执行阶段
task goodExample {
    println "Configuration phase: Task configured (no expensive operations)"

    // 延迟执行配置
    def expensiveResult = project.provider {
        expensiveOperation()
    }

    doLast {
        println "Execution phase: Performing expensive operation now"
        def result = expensiveResult.get()
        println "Result: $result"
    }
}

// 耗时操作模拟
def expensiveOperation() {
    println "Performing expensive operation (simulated)..."
    Thread.sleep(1000) // 模拟耗时操作
    return "Operation completed at ${new Date()}"
}

// 条件化配置示例
task conditionalConfiguration {
    println "Configuration phase: Task configured"

    // 条件化配置
    if (project.hasProperty('enableFeature')) {
        ext.featureEnabled = true
        println "Configuration phase: Feature enabled"
    } else {
        ext.featureEnabled = false
        println "Configuration phase: Feature disabled"
    }

    doLast {
        println "Execution phase: Feature is ${featureEnabled ? 'enabled' : 'disabled'}"
    }
}
```

### 3.2.3 Task依赖关系配置

**Task依赖关系类型：**

```groovy
// build.gradle - Task依赖关系配置

// 1. dependsOn - 强制依赖
task taskA {
    doLast {
        println "Executing Task A"
    }
}

task taskB {
    dependsOn taskA

    doLast {
        println "Executing Task B (depends on Task A)"
    }
}

task taskC {
    dependsOn taskB

    doLast {
        println "Executing Task C (depends on Task B)"
    }
}

// 2. mustRunAfter - 建议执行顺序（不强制依赖）
task taskD {
    doLast {
        println "Executing Task D"
    }
}

task taskE {
    doLast {
        println "Executing Task E"
    }
}

taskE.mustRunAfter taskD

// 3. shouldRunAfter - 弱建议执行顺序
task taskF {
    doLast {
        println "Executing Task F"
    }
}

task taskG {
    doLast {
        println "Executing Task G"
    }
}

taskG.shouldRunAfter taskF

// 4. finalizedBy - 最终化任务
task taskH {
    doLast {
        println "Executing Task H"
    }
}

task cleanup {
    doLast {
        println "Executing cleanup task"
    }
}

taskH.finalizedBy cleanup

// 5. 复杂依赖关系示例
task compileCode {
    doLast {
        println "Compiling source code"
    }
}

task runUnitTests {
    dependsOn compileCode

    doLast {
        println "Running unit tests"
    }
}

task runIntegrationTests {
    dependsOn compileCode

    doLast {
        println "Running integration tests"
    }
}

task generateTestReport {
    dependsOn runUnitTests, runIntegrationTests

    doLast {
        println "Generating test report"
    }
}

task deploy {
    dependsOn generateTestReport

    doLast {
        println "Deploying application"
    }

    finalizedBy {
        // 匿名最终化任务
        doLast {
            println "Deployment cleanup"
        }
    }
}

// 6. 动态依赖配置
task dynamicDependency {
    doLast {
        println "Task with dynamic dependencies"
    }
}

// 根据条件添加依赖
if (project.hasProperty('enableTests')) {
    dynamicDependency.dependsOn runUnitTests
}

if (project.hasProperty('enableIntegration')) {
    dynamicDependency.dependsOn runIntegrationTests
}

// 7. 任务依赖关系可视化
task showDependencyGraph {
    doLast {
        println "=== Task Dependency Graph ==="

        // 获取任务图
        def taskGraph = gradle.taskGraph

        if (taskGraph.executionPlan) {
            taskGraph.executionPlan.each { entry ->
                def task = entry.task
                def dependencies = entry.dependencies

                println "Task: ${task.path}"
                if (!dependencies.isEmpty()) {
                    println "  Depends on: ${dependencies.collect { it.task.path }.join(', ')}"
                }
                println ""
            }
        }
    }
}
```

**Task依赖关系最佳实践：**

```groovy
// build.gradle - Task依赖关系最佳实践

// 1. 生命周期阶段定义
// 定义标准生命周期阶段
task clean {
    description 'Clean build artifacts'
    group 'build'

    doLast {
        println "Cleaning build directory"
        delete buildDir
    }
}

task compile {
    description 'Compile source code'
    group 'build'
    dependsOn clean

    doLast {
        println "Compiling source code"
    }
}

task test {
    description 'Run tests'
    group 'verification'
    dependsOn compile

    doLast {
        println "Running tests"
    }
}

task build {
    description 'Build the project'
    group 'build'
    dependsOn test

    doLast {
        println "Building project"
    }
}

// 2. 功能模块任务组
task compileJava(type: JavaCompile) {
    description 'Compile Java sources'
    group 'java'

    source = sourceSets.main.allJava
    classpath = sourceSets.main.compileClasspath
    destinationDirectory = file("$buildDir/classes/java/main")
}

task compileTests(type: JavaCompile) {
    description 'Compile test sources'
    group 'java'
    dependsOn compileJava

    source = sourceSets.test.allJava
    classpath = sourceSets.test.compileClasspath
    destinationDirectory = file("$buildDir/classes/java/test")
}

task runUnitTests(type: Test) {
    description 'Run unit tests'
    group 'verification'
    dependsOn compileTests

    testClassesDirs = sourceSets.test.output.classesDirs
    classpath = sourceSets.test.runtimeClasspath
}

task runIntegrationTests(type: Test) {
    description 'Run integration tests'
    group 'verification'
    dependsOn jar

    testClassesDirs = sourceSets.test.output.classesDirs
    classpath = sourceSets.test.runtimeClasspath + jar.outputs.files

    include '**/*IntegrationTest.class'
}

// 3. 依赖关系优化
// 避免不必要的依赖
task fastTest(type: Test) {
    description 'Run fast tests only'
    group 'verification'
    dependsOn compileTests

    testClassesDirs = sourceSets.test.output.classesDirs
    classpath = sourceSets.test.runtimeClasspath

    exclude '**/*IntegrationTest.class'
    exclude '**/*SlowTest.class'
}

// 4. 条件化依赖
task conditionalBuild {
    description 'Conditional build based on properties'
    group 'build'

    doLast {
        println "Performing conditional build"
    }
}

// 根据属性添加依赖
if (project.hasProperty('runTests')) {
    conditionalBuild.dependsOn test
} else {
    conditionalBuild.dependsOn compile
}

// 5. 任务依赖关系验证
task validateDependencies {
    description 'Validate task dependencies'
    group 'verification'

    doLast {
        def issues = []

        // 检查循环依赖
        def visited = [] as Set
        def recursionStack = [] as Set

        gradle.taskGraph.allTasks.each { task ->
            if (hasCyclicDependency(task, visited, recursionStack)) {
                issues << "Cyclic dependency detected involving task: ${task.path}"
            }
        }

        // 检查缺失的依赖
        def requiredTasks = [compileJava, compileTests]
        requiredTasks.each { requiredTask ->
            if (!gradle.taskGraph.allTasks.contains(requiredTask)) {
                issues << "Required task not in execution graph: ${requiredTask.path}"
            }
        }

        if (issues) {
            println "Dependency validation issues found:"
            issues.each { println "  - $it" }
        } else {
            println "No dependency issues found"
        }
    }
}

// 检查循环依赖的辅助方法
def hasCyclicDependency(task, visited, recursionStack) {
    if (recursionStack.contains(task)) {
        return true
    }

    if (visited.contains(task)) {
        return false
    }

    visited.add(task)
    recursionStack.add(task)

    // 检查任务的依赖
    task.taskDependencies.getDependencies(task).each { dependency ->
        if (hasCyclicDependency(dependency, visited, recursionStack)) {
            return true
        }
    }

    recursionStack.remove(task)
    return false
}
```

### 3.2.4 Task输入输出规范

**Task输入输出声明：**

```groovy
// build.gradle - Task输入输出规范

// 1. 基础输入输出声明
task processFile {
    description 'Process a single file'
    group 'custom'

    // 输入声明
    @InputFile
    File inputFile = file('src/main/resources/input.txt')

    @Input
    String processingMode = 'uppercase'

    // 输出声明
    @OutputFile
    File outputFile = file("$buildDir/processed-output.txt")

    @OutputDirectory
    File outputDir = file("$buildDir/processed-files")

    doLast {
        println "Processing file: ${inputFile.name}"
        println "Processing mode: $processingMode"

        if (inputFile.exists()) {
            def content = inputFile.text
            def processedContent = processingMode == 'uppercase' ? content.toUpperCase() : content.toLowerCase()

            // 确保输出目录存在
            outputFile.parentFile.mkdirs()
            outputFile.text = processedContent

            println "Processed file written to: ${outputFile.absolutePath}"
        } else {
            throw new GradleException("Input file does not exist: ${inputFile.absolutePath}")
        }
    }
}

// 2. 文件集合输入输出
task processMultipleFiles {
    description 'Process multiple files'
    group 'custom'

    // 输入文件集合
    @InputFiles
    FileCollection inputFiles = files('src/main/resources').filter { it.name.endsWith('.txt') }

    @Input
    Map<String, String> processingOptions = [mode: 'uppercase', encoding: 'UTF-8']

    // 输出目录
    @OutputDirectory
    File outputDirectory = file("$buildDir/multi-processed")

    doLast {
        outputDirectory.mkdirs()

        inputFiles.each { file ->
            println "Processing file: ${file.name}"
            def content = file.getText(processingOptions.encoding)
            def processedContent = processingOptions.mode == 'uppercase' ? content.toUpperCase() : content.toLowerCase()

            def outputFile = new File(outputDirectory, "processed-${file.name}")
            outputFile.text = processedContent
        }

        println "Processed ${inputFiles.files.size()} files to ${outputDirectory.absolutePath}"
    }
}

// 3. 目录输入输出
task processDirectory {
    description 'Process entire directory'
    group 'custom'

    // 输入目录
    @InputDirectory
    File inputDirectory = file('src/main/resources')

    @Input
    boolean includeSubdirectories = true

    // 输出目录
    @OutputDirectory
    File outputDirectory = file("$buildDir/directory-processed")

    doLast {
        println "Processing directory: ${inputDirectory.absolutePath}"
        println "Include subdirectories: $includeSubdirectories"

        outputDirectory.mkdirs()

        // 复制和处理文件
        project.copy {
            from inputDirectory
            into outputDirectory
            include '**/*.properties'
            filter { line ->
                line.replace('\${project.version}', project.version.toString())
            }
        }

        println "Directory processing completed"
    }
}

// 4. 属性输入输出
task generateProperties {
    description 'Generate properties file'
    group 'custom'

    // 输入属性
    @Input
    String applicationName = 'My Application'

    @Input
    String version = project.version.toString()

    @Input
    Map<String, Object> additionalProperties = [:]

    // 输出文件
    @OutputFile
    File propertiesFile = file("$buildDir/generated.properties")

    doLast {
        println "Generating properties file"

        def properties = new Properties()
        properties['app.name'] = applicationName
        properties['app.version'] = version
        properties['build.timestamp'] = new Date().format('yyyy-MM-dd HH:mm:ss')
        properties['java.version'] = System.getProperty('java.version')

        additionalProperties.each { key, value ->
            properties[key] = value.toString()
        }

        propertiesFile.parentFile.mkdirs()
        propertiesFile.withOutputStream { stream ->
            properties.store(stream, "Generated by Gradle task")
        }

        println "Properties file generated: ${propertiesFile.absolutePath}"
    }
}

// 5. 自定义输入输出验证
task validatedProcessing {
    description 'Process with custom validation'
    group 'custom'

    // 自定义输入
    @Input
    String configValue

    @InputFiles
    FileCollection sourceFiles

    // 自定义输出
    @OutputFile
    File resultFile

    // 自定义验证方法
    @Inject
    ObjectFactory getObjectFactory() {
        throw new IllegalStateException("Cannot inject ObjectFactory")
    }

    // 输入验证
    @Internal
    boolean isValidInput() {
        return configValue != null && !configValue.trim().isEmpty() &&
               sourceFiles != null && !sourceFiles.isEmpty()
    }

    // 输出验证
    @Internal
    boolean isValidOutput() {
        return resultFile != null && resultFile.parentFile != null
    }

    doLast {
        if (!isValidInput()) {
            throw new GradleException("Invalid input: configValue and sourceFiles must be provided")
        }

        if (!isValidOutput()) {
            throw new GradleException("Invalid output: resultFile must be specified")
        }

        println "Processing with validated input/output"
        println "Config value: $configValue"
        println "Source files: ${sourceFiles.files.size()}"

        resultFile.parentFile.mkdirs()
        resultFile.text = "Processed with config: $configValue\nFiles processed: ${sourceFiles.files.size()}"

        println "Processing completed"
    }
}

// 6. 增量任务示例
task incrementalProcess(type: DefaultTask) {
    description 'Incremental processing task'
    group 'custom'

    // 增量输入
    @InputFiles
    @SkipWhenEmpty
    FileCollection inputFiles = fileTree('src/main/resources') {
        include '**/*.txt'
    }

    // 增量输出
    @OutputDirectory
    File outputDirectory = file("$buildDir/incremental-output")

    // 增量状态
    @Internal
    IncrementalTaskInputs incrementalInputs

    doLast {
        if (incrementalInputs.incremental) {
            println "Incremental processing: ${incrementalInputs.outOfDate.files.size()} out of date, ${incrementalInputs.removed.files.size()} removed"

            // 处理变更的文件
            incrementalInputs.outOfDate.files.each { file ->
                println "Processing changed file: ${file.name}"
                def outputFile = new File(outputDirectory, file.name)
                outputFile.text = "Processed at ${new Date()}: ${file.text}"
            }

            // 删除已移除的输出文件
            incrementalInputs.removed.files.each { file ->
                def outputFile = new File(outputDirectory, file.name)
                if (outputFile.exists()) {
                    outputFile.delete()
                    println "Removed output file: ${outputFile.name}"
                }
            }
        } else {
            println "Full processing (non-incremental)"
            outputDirectory.mkdirs()

            inputFiles.each { file ->
                println "Processing file: ${file.name}"
                def outputFile = new File(outputDirectory, file.name)
                outputFile.text = "Processed at ${new Date()}: ${file.text}"
            }
        }
    }
}

// 配置示例任务的任务
task setupExampleInputs {
    doLast {
        // 创建示例输入文件
        def inputDir = file('src/main/resources')
        inputDir.mkdirs()

        new File(inputDir, 'input1.txt').text = 'Hello World'
        new File(inputDir, 'input2.txt').text = 'Gradle Processing'
        new File(inputDir, 'config.properties').text = 'app.name=Demo App'
    }
}

// 设置示例任务的依赖
processFile.dependsOn setupExampleInputs
processMultipleFiles.dependsOn setupExampleInputs
processDirectory.dependsOn setupExampleInputs
incrementalProcess.dependsOn setupExampleInputs
```

### 3.2.5 自定义Task开发

**自定义Task开发完整示例：**

```groovy
// buildSrc/src/main/groovy/com/example/tasks/FileProcessorTask.groovy
package com.example.tasks

import org.gradle.api.DefaultTask
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileCollection
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.*
import org.gradle.work.InputChanges
import org.gradle.work.Incremental
import org.gradle.work.RemoveFiles

/**
 * 自定义文件处理任务
 * 支持增量处理和多种处理模式
 */
abstract class FileProcessorTask extends DefaultTask {

    // 使用新的抽象属性API (Gradle 5.6+)
    @InputDirectory
    @Incremental
    abstract DirectoryProperty getInputDirectory()

    @OutputDirectory
    abstract DirectoryProperty getOutputDirectory()

    @Input
    abstract Property<String> getProcessingMode()

    @Input
    @Optional
    abstract Property<String> getFileEncoding()

    @Input
    @Optional
    abstract Property<Boolean> getIncludeSubdirectories()

    @Input
    @Optional
    abstract Property<String> getFileFilter()

    @Input
    @Optional
    abstract Property<Boolean> getCreateBackup()

    @Console
    abstract Property<Boolean> getVerbose()

    // 构造函数
    FileProcessorTask() {
        // 设置默认值
        processingMode.convention('COPY')
        fileEncoding.convention('UTF-8')
        includeSubdirectories.convention(true)
        fileFilter.convention('**/*')
        createBackup.convention(false)
        verbose.convention(false)

        // 设置任务描述和分组
        group = 'file processing'
        description = 'Process files with various modes (COPY, TRANSFORM, COMPRESS)'

        // 设置输出目录
        outputDirectory.convention(project.layout.buildDirectory.dir('processed-files'))
    }

    /**
     * 增量处理逻辑
     */
    @TaskAction
    void execute(InputChanges inputChanges) {
        def outputDir = outputDirectory.get().asFile
        outputDir.mkdirs()

        if (inputChanges.incremental) {
            executeIncremental(inputChanges)
        } else {
            executeFull()
        }
    }

    /**
     * 增量执行
     */
    private void executeIncremental(InputChanges inputChanges) {
        def mode = processingMode.get()
        def verbose = verbose.get()

        if (verbose) {
            println "执行增量处理 - 模式: $mode"
            println "变更文件: ${inputChanges.outOfDate.files.size()}"
            println "移除文件: ${inputChanges.removed.files.size()}"
        }

        // 处理变更的文件
        inputChanges.outOfDate.each { change ->
            def inputFile = change.file
            def relativePath = inputDirectory.get().asFile.toPath().relativize(inputFile.toPath())
            def outputFile = outputDirectory.get().asFile.toPath().resolve(relativePath).toFile()

            processFile(inputFile, outputFile, mode)

            if (verbose) {
                println "处理变更文件: ${inputFile.name} -> ${outputFile.name}"
            }
        }

        // 处理移除的文件
        inputChanges.removed.each { change ->
            def inputFile = change.file
            def relativePath = inputDirectory.get().asFile.toPath().relativize(inputFile.toPath())
            def outputFile = outputDirectory.get().asFile.toPath().resolve(relativePath).toFile()

            if (outputFile.exists()) {
                if (createBackup.get()) {
                    def backupFile = new File("${outputFile}.backup")
                    outputFile.renameTo(backupFile)
                    if (verbose) {
                        println "备份并移除文件: ${outputFile.name} -> ${backupFile.name}"
                    }
                } else {
                    outputFile.delete()
                    if (verbose) {
                        println "移除文件: ${outputFile.name}"
                    }
                }
            }
        }
    }

    /**
     * 完整执行
     */
    private void executeFull() {
        def mode = processingMode.get()
        def verbose = verbose.get()
        def includeSubDirs = includeSubdirectories.get()
        def fileFilterPattern = fileFilter.get()

        if (verbose) {
            println "执行完整处理 - 模式: $mode"
            println "输入目录: ${inputDirectory.get()}"
            println "输出目录: ${outputDirectory.get()}"
        }

        // 清空输出目录
        project.delete(outputDirectory.get())
        outputDirectory.get().asFile.mkdirs()

        // 获取输入文件
        def inputFiles = project.fileTree(inputDirectory.get()) {
            include fileFilterPattern
            if (!includeSubDirs) {
                include '**/*'
                exclude '**/*/**' // 排除子目录
            }
        }

        def processedCount = 0
        inputFiles.each { inputFile ->
            def relativePath = inputDirectory.get().asFile.toPath().relativize(inputFile.toPath())
            def outputFile = outputDirectory.get().asFile.toPath().resolve(relativePath).toFile()

            // 确保输出目录存在
            outputFile.parentFile.mkdirs()

            processFile(inputFile, outputFile, mode)
            processedCount++

            if (verbose) {
                println "处理文件: ${inputFile.name} -> ${outputFile.name}"
            }
        }

        println "完整处理完成，共处理 $processedCount 个文件"
    }

    /**
     * 处理单个文件
     */
    private void processFile(File inputFile, File outputFile, String mode) {
        switch (mode.toUpperCase()) {
            case 'COPY':
                copyFile(inputFile, outputFile)
                break
            case 'TRANSFORM':
                transformFile(inputFile, outputFile)
                break
            case 'COMPRESS':
                compressFile(inputFile, outputFile)
                break
            default:
                throw new GradleException("不支持的处理模式: $mode")
        }
    }

    /**
     * 复制文件
     */
    private void copyFile(File inputFile, File outputFile) {
        project.copy {
            from inputFile
            into outputFile.parentFile
            rename { inputFile.name }
        }
    }

    /**
     * 转换文件
     */
    private void transformFile(File inputFile, File outputFile) {
        def encoding = fileEncoding.get()
        def content = inputFile.getText(encoding)

        // 执行转换逻辑
        def transformedContent = content
            .replaceAll('\${project.name}', project.name)
            .replaceAll('\${project.version}', project.version.toString())
            .replaceAll('\${build.timestamp}', new Date().format('yyyy-MM-dd HH:mm:ss'))

        outputFile.text = transformedContent
    }

    /**
     * 压缩文件
     */
    private void compressFile(File inputFile, File outputFile) {
        // 这里可以实现文件压缩逻辑
        // 为了示例，我们简单地将内容转换为一行
        def content = inputFile.getText(fileEncoding.get())
        def compressedContent = content.replaceAll(/\s+/, ' ').trim()

        outputFile.text = compressedContent
    }
}

// build.gradle - 使用自定义Task
task processStaticFiles(type: FileProcessorTask) {
    description 'Process static files for web application'
    group 'build'

    inputDirectory.set(layout.projectDirectory.dir('src/main/resources/static'))
    outputDirectory.set(layout.buildDirectory.dir('processed-static'))
    processingMode.set('TRANSFORM')
    fileEncoding.set('UTF-8')
    includeSubdirectories.set(true)
    fileFilter.set('**/*.{html,css,js}')
    verbose.set(true)
}

task processConfigFiles(type: FileProcessorTask) {
    description 'Process configuration files'
    group 'build'

    inputDirectory.set(layout.projectDirectory.dir('src/main/resources'))
    outputDirectory.set(layout.buildDirectory.dir('processed-config'))
    processingMode.set('COPY')
    fileFilter.set('**/*.{properties,yml,yaml}')
    createBackup.set(true)
    verbose.set(true)
}

// 链式配置示例
task processWithChain(type: FileProcessorTask) {
    inputDirectory.set(layout.projectDirectory.dir('src/main/resources'))
    processingMode.set('TRANSFORM')
    fileFilter.set('**/*.properties')

    doFirst {
        println "开始文件处理链"
    }

    doLast {
        println "文件处理链完成"

        // 可以继续配置其他任务
        if (project.hasProperty('enableCompression')) {
            processingMode.set('COMPRESS')
        }
    }
}
```

**自定义Task高级特性：**

```groovy
// buildSrc/src/main/groovy/com/example/tasks/AdvancedTask.groovy
package com.example.tasks

import org.gradle.api.DefaultTask
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.MapProperty
import org.gradle.api.provider.Property
import org.gradle.api.tasks.*
import org.gradle.workers.WorkAction
import org.gradle.workers.WorkParameters
import org.gradle.workers.WorkerExecutor
import javax.inject.Inject

/**
 * 支持并行处理的高级任务
 */
abstract class AdvancedFileProcessorTask extends DefaultTask {

    @InputFiles
    abstract ListProperty<File> getInputFiles()

    @OutputDirectory
    abstract DirectoryProperty getOutputDirectory()

    @Input
    abstract Property<String> getProcessingMode()

    @Input
    abstract Property<Integer> getMaxParallelWorkers()

    @Inject
    abstract WorkerExecutor getWorkerExecutor()

    AdvancedFileProcessorTask() {
        maxParallelWorkers.convention(Runtime.runtime.availableProcessors())
        group = 'advanced processing'
        description = 'Advanced file processor with parallel execution'
    }

    @TaskAction
    void execute() {
        def outputDir = outputDirectory.get().asFile
        outputDir.mkdirs()

        def inputFiles = getInputFiles().get()
        def maxWorkers = getMaxParallelWorkers().get()
        def mode = getProcessingMode().get()

        println "开始高级文件处理"
        println "文件数量: ${inputFiles.size()}"
        println "最大并行数: $maxWorkers"
        println "处理模式: $mode"

        // 将文件分组以便并行处理
        def chunkSize = Math.max(1, inputFiles.size() / maxWorkers)
        def fileChunks = inputFiles.collate(chunkSize)

        println "文件分组数: ${fileChunks.size()}"

        // 提交并行工作项
        fileChunks.eachWithIndex { chunk, index ->
            workerExecutor.noIsolation().submit(FileProcessorWorkAction.class) { parameters ->
                parameters.files.set(chunk)
                parameters.outputDirectory.set(outputDirectory.get().dir("chunk-$index"))
                parameters.processingMode.set(mode)
            }
        }
    }
}

/**
 * 工作操作接口
 */
interface FileProcessorParameters extends WorkParameters {
    ListProperty<File> getFiles()
    DirectoryProperty getOutputDirectory()
    Property<String> getProcessingMode()
}

/**
 * 具体的工作操作实现
 */
abstract class FileProcessorWorkAction implements WorkAction<FileProcessorParameters> {

    @Override
    void execute() {
        def files = parameters.files.get()
        def outputDir = parameters.outputDirectory.get().asFile
        def mode = parameters.processingMode.get()

        outputDir.mkdirs()

        println "工作线程处理 ${files.size()} 个文件"

        files.each { file ->
            def outputFile = new File(outputDir, file.name)
            processFile(file, outputFile, mode)
        }
    }

    private void processFile(File inputFile, File outputFile, String mode) {
        // 实现文件处理逻辑
        switch (mode) {
            case 'COPY':
                project.copy {
                    from inputFile
                    into outputFile.parentFile
                }
                break
            case 'TRANSFORM':
                def content = inputFile.text
                outputFile.text = content.toUpperCase()
                break
            default:
                outputFile.text = "Processed: ${inputFile.name}"
        }
    }
}

// build.gradle - 使用高级Task
task advancedFileProcessing(type: AdvancedFileProcessorTask) {
    description 'Advanced file processing with parallel execution'
    group 'advanced'

    inputFiles.set(files('src/main/resources').filter { it.name.endsWith('.properties') })
    outputDirectory.set(layout.buildDirectory.dir('advanced-processed'))
    processingMode.set('TRANSFORM')
    maxParallelWorkers.set(4)
}
```

通过本章的学习，您应该已经深入理解了Gradle的Project和Task系统，这是掌握Gradle构建机制的核心基础。在下一章中，我们将深入探讨Gradle强大的依赖管理系统，这对于企业级项目构建至关重要。