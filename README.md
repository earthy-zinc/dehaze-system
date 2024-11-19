## 项目介绍

基于深度学习的在线实时响应的图像去雾系统，主要功能是改善受到雾霾影响的图像质量，从⽽实现图像去雾的⽬标。采用 Vue3 + Vite4+ TypeScript5 + Element-Plus + Pinia 等最新主流技术栈构建。

详细介绍请见`./doc`文件夹。

## 系统亮点

1. **⽤户登录与注册：** 通过 websocket 定时请求确保⽤户登录状态，利⽤ Pinia 和浏览器 localStorage 持久化保存⽤户 Token、⻆⾊权限以及其他个性化设置。
2. **axios ⼆次封装：** 通过请求、响应拦截器拦截未登录、越权等⾮法请求，同时设计了⾼效的 API 接⼝代码结构和模块划分，提⾼了开发效率和可维护性。
3. **菜单展示：** 通过结合使⽤ VueRouter 和 Pinia，在登陆后动态获取⽤户菜单，对不同⽤户展示不同的操作菜单，实现静态路由和动态路由的分离。
4. **响应式布局：** 利⽤ CSS 变量、弹性盒⼦、相对单位，通过组件化管理 Vue 不同布局⻚⾯，使项⽬同时⽀持电脑端和移动端的显示，⽤同⼀套代码实现多种⻚⾯布局模式的切换，满⾜不同⽤户需求。
5. **图⽚上传：** 前端针对图⽚⼤⼩、后缀、MD5 进⾏校验，减轻后端服务器压⼒，实现极速上传，针对多图⽚通过并发请求，上传过程中利⽤进度条可视化上传流程，提⾼⽤户体验。
6. **⾼效数据集图⽚浏览：** 采⽤缩略图、瀑布流布局以及懒加载技术展示⼤规模数据集图⽚。将初始位置设置在屏幕之外，防⽌懒加载失效，在优化传输速度的同时确保⽤户体验
7. **图像重叠对⽐：** 利⽤ CSS clip-path 实现原图和算法效果图重叠对⽐，并通过可拖拽线条直观调整两图像之间⽐例，增强对⽐效果
8. **图像细节展示增强：** 基于 Canvas 实现图⽚放⼤镜功能，并通过滑块实时调整亮度对⽐度，进⼀步突出图像细节
9. **组件的封装和复⽤：** 将复杂组件如瀑布流、懒加载、放⼤镜进⾏封装，将逻辑抽离为 Hooks，提升代码重⽤性和可维护性。

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
