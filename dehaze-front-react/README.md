## 📢 项目介绍

基于深度学习的在线实时响应的图像去雾系统，主要功能是改善受到雾霾影响的图像质量，从⽽实现图像去雾的⽬标。

## 💻 技术栈

* 前端框架：React + TypeScript + Vite，采用现代前端开发工具链
* 状态管理：Redux Toolkit 实现模块化状态管理（store/modules下有多个slice）
* UI组件库：Ant Design 5.x 实现统一视觉风格
* 跨平台：Electron 31实现桌面端应用开发（desktop目录）
* 构建工具：Vite 5 + Unocss 实现快速开发和轻量级样式处理

## 🛞 系统功能

1. 用户管理模块：
   支持用户注册/登录/权限管理
   角色-权限-菜单三级权限控制（RoleAPI与AuthAPI）
   用户信息加密传输（Token认证机制）
2. 数据集管理：
   数据集分页展示（DatasetList组件）
   数据集详情页支持图片瀑布流展示（Waterfall组件）
   支持数据集导入导出（DatasetAPI.export接口）
3. 图像处理功能：
   实时摄像头捕获（Camera组件）
   图像叠加对比（OverlapImageShow组件）
   放大镜效果（Magnifier组件）
   图像参数调节（对比度/亮度控制）
4. 算法集成：
   算法工具栏支持参数配置（AlgorithmToolBar组件）
   模型选择与预测结果可视化（ModelAPI接口）

## 🚀 项目启动

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

## 🌺 项目部署

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
