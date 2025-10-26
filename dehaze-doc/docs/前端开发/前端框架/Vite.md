# Vite

## 概述

Vite是一个现代化的前端构建工具，包含开发服务器和构建指令功能。默认构建目标为支持以下特性的浏览器：
- 原生ESM语法的script标签
- 原生ESM动态导入
- import.meta

### 开发服务器

在开发期间，Vite作为服务器运行，index.html作为项目入口文件。Vite将index.html视为源码和模块图的一部分，能够解析：
- `<script type="module" src="...">`标签指向的JavaScript源码
- 内联引入的JavaScript代码
- 引用CSS的`<link href>`标签

URL路径处理：
- 以项目根目录为基础解析绝对路径
- 自动转换index.html中的URL
- 处理根目录外的依赖文件

### 启动方式

Vite项目可通过以下方式启动：
1. npm脚本中的vite命令
2. 直接使用npx vite运行
3. 运行`npx vite --help`获取完整命令行选项

## 核心功能

### npm依赖解析与预构建

Vite处理裸模块导入的流程：
1. 检测源文件中的裸模块导入
2. 预构建依赖以提高页面加载速度
3. 将CommonJS模块转换为ES模块格式
4. 重写导入为合法URL供浏览器正确导入

### 模块热替换

Vite支持模块热替换(Hot Module Replacement, HMR)，在开发过程中实现局部更新，提升开发体验。

### TypeScript支持

Vite原生支持TypeScript，可直接导入和编译TypeScript文件。

### 静态资源处理

导入静态资源会返回解析后的URL：

```js
import imgUrl from './img.png'
console.log(imgUrl)
```

JSON文件也可直接导入使用。