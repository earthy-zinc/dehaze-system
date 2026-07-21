# dehaze-front-react

基于深度学习的图像去雾系统 React 前端。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/React前端架构文档.md)。

## 技术栈

- React + TypeScript + Vite 5
- Redux Toolkit（模块化状态管理）
- Ant Design 5.x
- Electron 31（桌面端，desktop 目录）
- UnoCSS

## 快速开始

```bash
# 安装 pnpm
npm install pnpm -g

# 设置环境变量
set ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/
set ELECTRON_BUILDER_BINARIES_MIRROR=https://npmmirror.com/mirrors/electron-builder-binaries/

# 安装依赖
PNPM_APPROVE_BUILDS=1 pnpm install

# 启动运行
pnpm run dev
```

## 部署

```bash
# 项目打包
pnpm run build:prod

# 上传文件至远程服务器
将打包生成在 `dist` 目录下的文件拷贝至 `/usr/share/nginx/html` 目录

# nginx.config 配置
server {
    listen     80;
    server_name  localhost;
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
    }
}
```
