# Jenkins 练习

持续交付流水线，这种流水线定义被写在了一个文本文件jenkinsfile。被提交到项目的源代码中。

这样的话有以下一些好处：

* 自动为所有分支创建流水线构建过程并拉取请求
* 在流水线上进行代码复查和迭代
* 对流水线进行审计和跟踪

## Jenkinsfile

```jenkinsfile
pipeline {
	agent any
	stages {
		stage('build'){
			steps{...
		}
	}
}
```

* pipeline是声明式流水线的一种特定语法，定义包含执行整个流水线的所有内容和指令块
* agent指示Jenkins为整个流水线分配什么执行器和工作区
* stage是描述流水线的一个阶段，一般软件部署的阶段。
* steps描述了在某个阶段上要执行的步骤
* sh 是shell命令
* junit是Java单元测试的流水线阶段