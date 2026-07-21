# dehaze-taro

基于 Taro 框架的图像去雾系统多端应用，支持编译到微信小程序、H5、支付宝小程序等平台。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/Taro前端架构文档.md)。

## 技术栈

- Taro 4.1.4
- React 18
- TypeScript
- Redux（状态管理）
- Less（样式预处理）

## 环境要求

- Node.js >= 18
- pnpm >= 8

## 快速开始

```bash
# 安装依赖
pnpm install
```

### 多端运行命令

| 平台 | 开发模式 | 生产构建 |
|------|---------|---------|
| 微信小程序 | `pnpm dev:weapp` | `pnpm build:weapp` |
| H5 网页 | `pnpm dev:h5` | `pnpm build:h5` |
| 支付宝小程序 | `pnpm dev:alipay` | `pnpm build:alipay` |
| 百度小程序 | `pnpm dev:swan` | `pnpm build:swan` |
| 头条小程序 | `pnpm dev:tt` | `pnpm build:tt` |
| QQ 小程序 | `pnpm dev:qq` | `pnpm build:qq` |
| 京东小程序 | `pnpm dev:jd` | `pnpm build:jd` |
| 快应用 | `pnpm dev:quickapp` | `pnpm build:quickapp` |
