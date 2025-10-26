# Maven

## 概述

Maven 是一个 Java 项目的管理和构建工具，主要功能包括：

- 使用 pom.xml 定义项目内容，并采用预设的目录结构
- 声明依赖项后自动下载并导入到 classpath（项目路径中）
- 使用 `groupId:artifactId:version` 唯一定位依赖（公司名称:项目名称:版本号）

### 项目目录结构

一个 Maven 项目的标准目录结构如下：

```ascii
a-maven-project                 项目名称
├── pom.xml                     项目的描述文件
├── src                         项目源代码存放处
│   ├── main        
│   │   ├── java                存放Java源代码
│   │   └── resources           存放资源文件的目录
│   └── test
│       ├── java                存放测试用的代码
│       └── resources           存放测试用的资源
└── target                      所有编译、打包生成的文件
```

## 依赖管理

Maven 解决了依赖管理的问题，当项目依赖某个文件，而该文件又依赖其他文件时，Maven 可以自动下载并配置整个依赖链。

### 依赖范围

Maven 定义了四种依赖范围：

| 范围       | 说明                                       |
|------------|--------------------------------------------|
| compile    | （默认情况）编译时需要用到                 |
| test       | 编译测试文件时需要用到                     |
| runtime    | 运行时需要，编译时不需要                   |
| provided   | 编译时需要用到，但运行时由 JDK 或其他服务器提供 |

## POM 文件

pom.xml 存放在 Maven 管理的项目根目录中，包含：

- 当前项目的信息
- 用于构建编译项目的各种配置详细信息
- 项目执行目标和插件

在执行编译部署等任务时，Maven 从当前目录中查找 pom.xml，获取所需的配置信息。

### 父 POM

Super pom.xml 文件，也称为超级 POM 或父 POM，是 Maven 项目的默认 POM 配置，所有项目都默认继承自该 POM 文件。

我们自己配置的 POM 只包含指定的配置，不包含从父类继承的 Maven 配置。使用 `mvn help:effective-pom` 命令可以查看当前项目包含的默认 POM 文件的全部配置。

## 构建生命周期

构建生命周期（Build Life Cycle）指项目从编译到构建为字节码文件的整个流程。

### 默认生命周期

| 阶段                      | 描述                                       |
|---------------------------|--------------------------------------------|
| validate                  | 验证项目是否正确，检查所有必要的信息       |
| initialize                | 初始化构建状态                             |
| generate-sources          | 生成编译阶段及之后阶段需要的所有源代码     |
| process-sources           | 处理源代码                                 |
| generate-resources        | 生成要包含在软件包中的资源文件             |
| process-resources         | 将资源复制到目标目录，为打包阶段做好准备   |
| compile                   | 编译源代码                                 |
| process-classes           | 对编译后生成的类文件进行处理，如字节码增强 |
| generate-test-sources     | 生成测试源代码                             |
| process-test-sources      | 处理测试源代码                             |
| test-compile              | 编译测试代码                               |
| process-test-classes      | 处理编译后的测试类文件                     |
| test                      | 使用合适的测试框架对代码进行测试           |
| prepare-package           | 在实际打包之前需要进行的操作               |
| package                   | 将代码打包为 jar、war 等文件包             |
| pre-integration-test      | 执行集成测试之前需要的操作                 |
| integration-test          | 将程序包部署到可运行集成测试的环境中并测试 |
| post-integration-test     | 集成测试后需要的操作，如清理环境           |
| verify                    | 运行检查验证包是否有效并符合质量标准       |
| install                   | 将软件安装到本地仓库                       |
| deploy                    | 将软件复制到远程仓库                       |

当通过 Maven 命令调用某个阶段时，Maven 会运行该阶段及之前的所有阶段。根据软件包类型（jar/war），不同的构建目标会采用不同的阶段。

## 插件

Maven 本身只是一个容纳插件的容器，每个构建任务都由插件完成，这些插件通常用于：

- 编译源代码
- 单元测试
- 构建项目文档
- 创建项目构建成果报告
- 将源代码打包为 jar/war 文件

插件提供许多构建目标，可通过 mvn 命令指定插件执行特定构建目标，通用语法为：
```bash
mvn [plugin-name]:[goal-name]
```

### 插件类型

1. 构建项目插件
2. 创建报告插件

## 外部依赖

当远程仓库和中央仓库都没有项目所需的依赖包时，可使用 Maven 的外部依赖功能管理本地依赖文件：

```xml
<dependency>
  <groupId>ldapjdk</groupId>
  <artifactId>ldapjdk</artifactId>
  <!-- 将范围指定为本地系统 -->
  <scope>system</scope>
  <version>1.0</version>
  <!-- 指定依赖包相对于项目位置的系统路径 -->
  <systemPath>${basedir}\src\lib\ldapjdk.jar</systemPath>
</dependency>
```

## 项目文档网站

Maven 通过 site 插件为项目生成说明文档的静态网页，位于项目目录的 target/site 文件夹下，可部署到服务器上。

## 项目模板

Maven 提供多个项目模板，帮助用户快速创建各种类型的 Java 项目。使用 archetype 插件根据模板创建项目结构。

## 自动化构建

对于有复杂依赖关系的多个项目，Maven 支持自动构建，无需每次手动指定构建顺序。解决方案包括：

1. 整理项目依赖关系，对依赖其他项目的项目添加 post-build 目标，在构建前先构建所依赖的项目
2. 通过持续集成工具自动管理构建
3. 采用父项目聚合所有子项目并规定构建顺序