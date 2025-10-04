# 去雾系统 - Taro 多端版本

基于 Taro 框架开发的图像去雾系统，支持编译到微信小程序、H5、支付宝小程序等多个平台。

## 项目介绍

这是一个使用 Taro 框架构建的多端应用程序，专门用于图像去雾处理。该应用允许用户登录/注册账户，上传图片，并通过深度学习算法对雾霾图像进行处理，以获得更清晰的图像效果。支持编译到多个平台，包括微信小程序、H5、支付宝小程序等。

## 技术栈

- Taro 4.1.4
- React 18
- TypeScript
- Redux (状态管理)
- Less (样式预处理)

## 环境要求

- Node.js >= 18
- pnpm >= 8

## 安装步骤

1. 克隆项目代码:
   ```
   git clone <repository-url>
   ```

2. 进入项目目录:
   ```
   cd dehaze-taro
   ```

3. 安装依赖:
   ```
   pnpm install
   ```

## 运行项目

### 微信小程序

开发模式:
```
pnpm dev:weapp
```

构建生产版本:
```
pnpm build:weapp
```

### H5 网页

开发模式:
```
pnpm dev:h5
```

构建生产版本:
```
pnpm build:h5
```

### 其他平台

支付宝小程序:
```
pnpm dev:alipay
pnpm build:alipay
```

百度小程序:
```
pnpm dev:swan
pnpm build:swan
```

头条小程序:
```
pnpm dev:tt
pnpm build:tt
```

QQ 小程序:
```
pnpm dev:qq
pnpm build:qq
```

京东小程序:
```
pnpm dev:jd
pnpm build:jd
```

快应用:
```
pnpm dev:quickapp
pnpm build:quickapp
```

## 项目结构

```
dehaze-taro/
├── config/                  # Taro 配置文件
├── src/                     # 应用源代码
│   ├── app.config.ts        # 应用配置文件
│   ├── app.tsx              # 应用入口文件
│   ├── index.html           # H5 模板文件
│   └── pages/               # 页面组件
│       └── login/           # 登录/注册页面
├── types/                   # TypeScript 类型定义
├── package.json             # 项目依赖和脚本配置
└── ...
```

## 功能特性

### 登录/注册页面
- 用户登录功能
- 用户注册功能
- 表单验证
- 响应式设计，适配手机和平板

## 开发说明

### UI 设计规范
- 采用现代化、简约大方的设计风格
- 适配移动端竖屏显示，布局紧凑
- 元素尺寸和间距适合手指操作

### 页面适配
- 登录/注册页面针对手机竖屏进行了优化
- 减少垂直空间占用，避免布局松散
- 平板端有专门的适配样式

## 未来规划

1. 集成图像上传功能
2. 实现图像去雾核心算法
3. 添加图像处理历史记录
4. 增加更多页面和功能模块

## 许可证

[待定]