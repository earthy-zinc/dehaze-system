# Maven

## Maven介绍
Maven是一个Java项目的管理和构建工具。

* Maven使用`pom.xml`定义项目的内容，并且使用预设的目录结构。
* 在Maven中声明一个依赖项就可以自动下载并导入到`classpath`（项目路径中）。
* Maven使用`groupId artifactId version`唯一定位一个依赖（公司名称，项目名称，版本号）。

一个Maven项目的目录结构如下：

```ascii
a-maven-project				 项目名称
├── pom.xml					项目的描述文件
├── src				  		项目源代码存放处
│   ├── main		
│   │   ├── java		 	存放Java源代码
│   │   └── resources		 存放资源文件的目录
│   └── test
│       ├── java		 	 存放测试用的代码
│       └── resources	 	 存放测试用的资源
└── target					所有编译、打包生成的文件
```

**依赖管理**

Maven解决了依赖管理的问题，比如说我们的项目依赖一个文件，而这个文件又依赖另一些文件，那么Maven就可以把这一连串依赖的文件帮我们下载并配置号。

**依赖关系**

Maven定义了几种依赖关系，分别是编译中、测试时、运行中、提供四种依赖关系

| 范围       | 说明                         |
|----------|----------------------------|
| compile  | （默认情况）编译时需要用到              |
| test     | 编译测试文件时需要用到                |
| runtime  | 运行时需要，编译时不需要               |
| provided | 编译时需要用到，但是运行时由JDK或者其他服务器提供 |

## Pom.xml

* pom.xml存放在由maven管理的项目文件根目录中。
* pom.xml中包含当前项目的信息以及用于构建编译项目的各种配置的详细信息。
* pom.xml包含项目执行目标和插件，在执行编译部署等任务时，Maven从当前目录中查找pom.xml，获取所需的配置信息。

### 父类POM

Super Pom.xml文件，又叫超级Pom，父类pom，是Maven项目的默认pom配置，所得的项目都默认继承自该pom文件。
我们自己配置的pom只包含自己指定的配置，而不包含我们从父类继承的maven配置。
我们使用`mvn help:effective-pom`命令可以查看当前项目包含了默认pom文件的全部配置。

## 构建生命周期

构建生命周期(Build Life Cycle)指的是项目从编译到构建为字节码文件的整个流程。整个构建阶段的流程可以分为以下几个部分

| 阶段                | 描述                        |
|-------------------|---------------------------|
| prepare-resources | 资源复制阶段，在此阶段自定义资源复制到哪个地方   |
| validate          | 验证信息，验证项目是否正确，所有必要的信息是否可用 |
| compile           | 源代码编译阶段                   |
| Test              | 测试代码                      |
| package           | 打包项目的源代码，创建jar/war包       |
| install           | 将程序包安装到本地或者远程的maven仓库     |
| deploy            | 运行部署程序                    |

## 默认生命周期

| 阶段                    | 描述                          | 
|-----------------------|-----------------------------|
| validate              | 验证项目是否正确，检查所有必要的信息          |
| initialize            | 初始化构建状态                     |
| generate-sources      | 生成编译阶段及之后阶段需要的所有源代码         |
| process-sources       | 处理源代码                       |
| generate-resources    | 生成要包含在软件包中的资源文件             |
| process-resources     | 将资源复制到目标目录，为打包阶段做好准备        |
| compile               | 编译源代码                       |
| process-classes       | 对编译后生成的类文件进行处理，例如对字节码增强和优化  |
| generate-test-sources | 生成测试源代码                     |
| process-test-sources  |                             |
| test-compile          |                             |
| process-test-classes  |                             |
| test                  | 使用合适的测试框架对代码进行测试            |
| prepare-package       | 在实际对代码打包之前需要进行的操作           |
| package               | 将代码打包为jar、war等文件包           |
| pre-integration-test  | 执行集成测试之前需要的操作               |
| integration-test      | 讲程序包部署到可运行集成测试的环境中，并且进行集成测试 |
| post-integration-test | 集成测试后需要的操作，如清理环境            |
| verify                | 运行任何检查验证包是否有效并符合质量标准        |
| install               | 将软件安装到本地仓库                  |
| deploy                | 将软件复制到远程仓库                  |

* 当通过maven命令调用整个流程的某一个阶段时，maven会运行该阶段以及之前的所有阶段。
* 根据软件包的类型jar/war，不同的maven构建目标会采用maven流程的不同阶段

## 站点生命周期

site阶段，用于？

## 构建流程的配置

构建（build）项是是一组配置值。制定了项目采用maven构建整个流程的各种配置。
在不同环境下我们可以使用profiles使用不同的构建配置。这样maven就可以在不同环境下采用不同的配置。

## 构建的配置文件

构建配置文件有三种类型
在每个项目的根目录pom.xml中
每个用户中%USER_HOME%/.m2/setting.xml
每台计算机全局配置%M2_HOME%/conf/setting.xml

我们可以通过多种方式激活maven的配置文件，默认情况下，maven配置文件激活的顺序是项目——用户——全局

* 在控制台通过命令激活
* 通过maven设置
* 基于环境变量
* 操作系统的设置

## maven插件

maven本身只是一个容纳插件的容器，每个构建任务都是由插件完成的，
这些插件通常用于编译源代码、单元测试、构建项目文档、创建项目构建成果报告、
将源代码打包为jar/war文件等任务。

插件本身提供了许多构建目标，我们可以通过mvn命令指定某个插件，执行某项构建目标。
通用的语法如下所示：
mvn [plugin-name]:[goal-name]

### maven插件类型

1. 构建项目的插件
2. 创建报告的插件

## maven外部依赖包

如果任何远程仓库和中央仓库都没有项目对应的依赖包，
那么我们就需要maven为我们提供的外部依赖包功能。
这样的话maven就能为我们管理本地依赖文件。

```xml

<dependency>
  <groupId>ldapjdk</groupId>
  <artifactId>ldapjdk</artifactId>
  <!--  将范围指定为本地系统-->
  <scope>system</scope>
  <version>1.0</version>
  <!--  指定依赖包相对于项目位置的系统路径-->
  <systemPath>${basedir}\src\lib\ldapjdk.jar</systemPath>
</dependency>
```

## maven创建项目文档网站

maven通过site插件能够为项目生成说明文档的静态网页。在项目目录的target/site文件夹下。随后我们可以将这些静态网页部署到服务器上。

## maven项目模板

maven为用户提供了多个项目模板，帮助用户快速创建各种不同类型的Java项目。
maven使用archetype插件来完成这个功能，这个插件就能根据模板来创建项目结构。

## maven自动化构建

假设我们有多个项目，其中一个项目依赖另一个或多个项目，这几个项目之间有复杂的依赖关系，我们如果想要构建这样的项目，就会比较费时费力，在构建时就需要
人工指定顺序，而现在maven能帮助我们自动构建这些项目，而不必每次构建时都需要关心他们之间的依赖关系。
对于这种复杂依赖的多个项目，我们可以采用以下三种方式解决：

1. 在构建项目之前整理清晰每个项目的依赖关系，对于依赖其他项目的项目，添加一个post-build目标。在构建该项目之前先构建其所以来的项目
2. 通过持续集成工具自动管理构建
3. 采用父项目聚合所有子项目并规定构建顺序

## maven依赖管理
