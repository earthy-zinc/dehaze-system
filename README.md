## 项目介绍

基于 Vue3 + Vite4+ TypeScript5 + Element-Plus + Pinia 等最新主流技术栈构建的图像去雾系统前端。系统详细介绍请见`./doc`文件夹


## 环境准备

| 环境               | 名称版本                                                                                            | 备注                                                                                        |
|------------------|:------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **开发工具**         | VSCode                                                                                          | [下载地址](https://code.visualstudio.com/Download)                                            |
| **运行环境**         | Node 16+                                                                                        | [下载地址](http://nodejs.cn/download)                                                         |
| **VSCode插件(必装)** | 1. `Vue Language Features (Volar) ` <br/> 2. `TypeScript Vue Plugin (Volar) `  <br/>3. 禁用 Vetur | ![vscode-plugin](https://foruda.gitee.com/images/1687755823108948048/d0198b2d_716974.png) |


## 项目启动

```bash
# 克隆代码
git clone https://gitee.com/earthy-zinc/dehaze_front.git

# 切换目录
cd dehaze_front

# 安装 pnpm
npm install pnpm -g

# 安装依赖
pnpm install

# 启动运行
pnpm run dev
```

## 项目部署

```bash
# 项目打包
pnpm run build:prod

# 上传文件至远程服务器
将打包生成在 `dist` 目录下的文件拷贝至 `/usr/share/nginx/html` 目录

# nginx.cofig 配置
server {
	listen     80;
	server_name  localhost;
	location / {
			root /usr/share/nginx/html;
			index index.html index.htm;
	}
}
```
