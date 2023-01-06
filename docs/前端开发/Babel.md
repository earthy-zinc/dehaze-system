# Babel 练习

## 一、Babel介绍

Babel是一个JavaScript编译器，可以将那些用ES2015及以上语法写的JavaScript代码转换为低版本的JavaScript代码，以便能够运行在旧版本的浏览器及环境中。

如果需要Babel工作，我们需要在项目的根目录下安装Babel相关的软件包，并且创建Babel的配置文件babel.config.json或者babel.config.js（旧版本），然后我们可以运行babel的命令将项目中所有JS代码编译到指定的目录中。

Babel软件包涉及到的模块：

* 核心库 core lib：Babel的核心功能都包含在 [@babel/core](https://www.babeljs.cn/docs/babel-core) 模块中
* 命令行工具 cli tool：[@babel/cli](https://www.babeljs.cn/docs/babel-cli) 模块可以让你从终端中通过命令行使用Babel的相关功能
* 插件和预设 plugin and preset ：Babel中的代码转换功能是通过插件的形式出现的，我们可以通过引入多个插件将高版本的JavaScript代码转换为低版本的JavaScript代码。而@babel/preset-env 模块 整合了一组与代码转换相关的插件，我们可以通过安装它来引入所有的代码转换功能。