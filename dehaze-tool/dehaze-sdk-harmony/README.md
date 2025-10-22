# Dehaze SDK Harmony

Dehaze SDK Harmony 是一个基于 ArkTS 开发的鸿蒙原生 SDK，用于与去雾系统后端 API 进行交互。

## 功能特点

- 基于 ArkTS 开发，适配鸿蒙系统
- 封装了系统的各种 API 接口调用
- 支持 Token 认证机制
- 使用鸿蒙原生网络请求能力
- 自动处理请求和响应拦截

## 安装

将本SDK文件夹复制到鸿蒙项目的 libs 目录下，并在项目的 oh-package.json5 中添加依赖：

```json
{
  "dependencies": {
    "dehaze-sdk-harmony": "file:./libs/dehaze-sdk-harmony"
  }
}
```

然后执行 `ohpm install` 命令安装依赖。

## 使用方法

### 初始化 SDK

```typescript
import { DehazeSDK } from 'dehaze-sdk-harmony'

// 初始化 SDK
DehazeSDK.initialize(new DehazeSDK.Builder()
  .setBaseUrl('http://localhost:8989')
  .setDebug(true));
```

或者使用更简洁的方式：

```typescript
import { DehazeSDK } from 'dehaze-sdk-harmony'

// 初始化 SDK
DehazeSDK.initialize(
  new DehazeSDK.Builder()
    .setBaseUrl('http://localhost:8989')
    .setDebug(true)
    .build()
);
```

### 基本用法

```typescript
import { UserAPI } from 'dehaze-sdk-harmony'

// 获取用户信息
UserAPI.getInfo().then(result => {
  console.log(result);
});

// 获取用户分页列表
UserAPI.getPage({
  pageNum: 1,
  pageSize: 10
}).then(result => {
  console.log(result);
});
```

### 认证

SDK 使用鸿蒙系统提供的数据存储能力存储认证 Token，默认键名为 `accessToken`。您只需确保在用户登录后将 Token 存储在正确的位置即可。

```typescript
// 用户登录后设置 Token
import { TokenManager } from 'dehaze-sdk-harmony'
TokenManager.setToken('your-jwt-token');
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