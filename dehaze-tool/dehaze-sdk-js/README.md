# Dehaze SDK JS

Dehaze SDK JS 是一个基于 TypeScript 开发的 JavaScript SDK，用于与去雾系统后端 API 进行交互。

## 功能特点

- 基于 TypeScript 开发，提供完整的类型定义
- 封装了系统的各种 API 接口调用
- 支持 Token 认证机制
- 使用 Axios 作为 HTTP 客户端
- 自动处理请求和响应拦截

## 安装

### 本地使用（推荐）

在项目根目录下，使用相对路径安装：

```bash
pnpm add file:../dehaze-tool/dehaze-sdk-js
```

### 构建项目

如果需要重新构建 SDK：

```bash
# 进入 SDK 目录
cd ../dehaze-tool/dehaze-sdk-js

# 安装依赖
pnpm install

# 构建项目
pnpm build
```

## 使用方法

### 基本用法

```typescript
import { UserAPI } from 'dehaze-sdk-js';

// 获取用户信息
UserAPI.getInfo().then(userInfo => {
  console.log(userInfo);
});

// 获取用户分页列表
UserAPI.getPage({
  pageNum: 1,
  pageSize: 10
}).then(pageData => {
  console.log(pageData);
});
```

### 认证

SDK 使用 localStorage 存储认证 Token，默认键名为 `accessToken`。您只需确保在用户登录后将 Token 存储在正确的位置即可。

```typescript
// 用户登录后设置 Token
localStorage.setItem('accessToken', 'your-jwt-token');
```

### API 列表

SDK 当前包含以下 API 模块：

- AlgorithmAPI - 算法相关接口
- AuthAPI - 认证相关接口
- DatasetAPI - 数据集相关接口
- DeptAPI - 部门相关接口
- DictAPI - 字典相关接口
- FileAPI - 文件相关接口
- MenuAPI - 菜单相关接口
- ModelAPI - 模型相关接口
- RoleAPI - 角色相关接口
- UserAPI - 用户相关接口

## 开发指南

### 项目结构

```
src/
├── api/           # 各个模块的 API 封装
├── enums/         # 枚举类型定义
├── types/         # 全局类型定义
└── utils/         # 工具函数
```

### 构建

```bash
# 清理构建产物
pnpm clean

# 构建项目
pnpm build
```

构建后的文件将输出到 `dist` 目录。

## 类型定义

所有接口都有完整的 TypeScript 类型定义，可以在开发过程中提供完整的类型检查和智能提示。

## 浏览器兼容性

由于使用了较新的 JavaScript 特性，建议在现代浏览器中使用。如需支持旧版浏览器，请配置相应的 polyfill。

## 许可证

ISC
