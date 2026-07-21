# dehaze-front-vue

基于深度学习的图像去雾系统 Vue3 前端。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Vue前端架构文档.md)。

## 技术栈

- Vue 3.5 + Vite 7 + TypeScript 5
- Element Plus 2.13 + Pinia 3.0 + Vue Router 4.6
- UnoCSS + ECharts 6.0 + SockJS/StompJS (WebSocket)
- ESLint + Prettier + Stylelint + Husky

## 快速开始

```bash
# 安装 pnpm
npm install pnpm -g

# 安装依赖
pnpm install

# 启动开发服务
pnpm run dev
```

访问: http://localhost:5174

| 命令 | 说明 |
|------|------|
| `pnpm run build` | 生产构建 |
| `pnpm run test:unit` | 单元测试 (Vitest) |
| `pnpm run test:e2e` | E2E 测试 (Playwright) |
| `pnpm run lint` | 代码检查 |
| `pnpm run storybook` | Storybook 组件开发 |
| `pnpm run dev:electron` | Electron 桌面端开发 |

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
