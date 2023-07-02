# Vite 学习

## 介绍

Vite是一个前端构建工具，它有开发服务器、构建指令的功能。默认的构建目标是能支持原生ESM语法的script标签、原生ESM动态导入和import.meta的浏览器。

在开发期间Vite是一个服务器，而index.html是该Vite项目的入口文件。Vite将这个文件视为源码和模块图的一部分，Vite解析`<script type="module" src="...">`这个标签会指向JavaScript源码，甚至内联引入JavaScript的`<script type="module">`和引用Css的`<link herf>`也能利用Vite特有的功能被解析，index.html中的URL会被自动转换。

与静态HTTP服务器类似，Vite也需要根目录，即服务文件的位置，源码中的绝对URL路径都会是以项目的根作为基础解析。Vite还能够处理依赖关系，解析处于根目录外的文件位置。vite会以当前工作目录作为根目录启动开发服务器。同时解析项目根目录下的配置文件。

vite项目可以在npm脚本中使用vite命令，或者直接使用npx vite运行命令。运行`npx vite --help`能获得完整命令行选项。

## 功能

### npm依赖解析和与构建

对于裸模块导入，Vite会检测所有被加载的源文件中裸模块导入，会预构建他们，以提高页面加载速度，并且将CommonJS模块转换为ES模块格式。重写导入为合法的url，以便浏览器能够正确导入。

### 模块热替换

### typescript支持

### 处理静态资源

导入一个静态资源会返回解析后的url，如

```js
import imgUrl from './img.png'
console.log(imgUrl)
```

JSON也可以为直接导入

