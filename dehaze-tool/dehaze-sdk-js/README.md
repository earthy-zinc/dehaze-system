# Dehaze SDK for JavaScript

JavaScript SDK for Dehaze System API

## 安装

```bash
pnpm add dehaze-sdk-js
```

或者使用 npm:

```bash
npm install dehaze-sdk-js
```

## 构建

```bash
pnpm run build
```

## 配置

在使用 SDK 之前，您可能需要配置基础 URL 和其他 axios 设置。SDK 提供了四个拦截点：请求前、请求错误、响应和响应错误。

```javascript
import { configJavaAxios, configPythonAxios } from 'dehaze-sdk-js';

// 配置 Java 后端 API
configJavaAxios({
  onRequest: (config) => {
    // 修改请求配置
    config.baseURL = 'http://localhost:8989';
    config.timeout = 5000;
    return config;
  },
  onRequestError: (error) => {
    // 处理请求错误
    console.error('Request error:', error);
    return Promise.reject(error);
  },
  onResponse: (response) => {
    // 处理响应数据
    return response.data;
  },
  onResponseError: (error) => {
    // 处理响应错误
    console.error('Response error:', error);
    return Promise.reject(error);
  }
});

// 配置 Python 后端 API
configPythonAxios({
  onRequest: (config) => {
    // 修改请求配置
    config.baseURL = 'http://localhost:8991';
    return config;
  },
  onRequestError: (error) => {
    // 处理请求错误
    console.error('Python API request error:', error);
    return Promise.reject(error);
  },
  onResponse: (response) => {
    // 处理响应数据
    return response.data;
  },
  onResponseError: (error) => {
    // 处理响应错误
    console.error('Python API response error:', error);
    return Promise.reject(error);
  }
});
```

四个配置项说明：

- `onRequest`: 请求前拦截器，类型为 `(config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig`，可用于修改请求配置，如设置
  baseURL、timeout 等
- `onRequestError`: 请求错误拦截器，类型为 `(error: AxiosError) => any`，处理请求发送失败的情况
- `onResponse`: 响应拦截器，类型为 `(response: AxiosResponse) => any`，处理正常响应的数据
- `onResponseError`: 响应错误拦截器，类型为 `(error: AxiosError) => any`，处理响应异常的情况，如网络错误、HTTP 状态码错误等

## 使用方法

```javascript
import { UserAPI } from 'dehaze-sdk-js';

UserAPI.getInfo().then(res => {
  console.log(res);
});
```

或者使用解构导入:

```javascript
import { UserAPI, RoleAPI, MenuAPI } from 'dehaze-sdk-js';

const users = await UserAPI.getPage({ pageNum: 1, pageSize: 10 });
const roles = await RoleAPI.getOptions();
const menus = await MenuAPI.getList({});
```

## API 模块

SDK 导出以下 API 模块，详细方法签名请参考源码 `src/api/` 目录：

| 模块 | 导入名 | 说明 |
|------|--------|------|
| 认证 | `AuthAPI` | 登录、登出、验证码 |
| 用户 | `UserAPI` | 用户 CRUD、导入导出、密码管理 |
| 角色 | `RoleAPI` | 角色 CRUD、菜单权限分配 |
| 菜单 | `MenuAPI` | 菜单/路由 CRUD |
| 字典 | `DictAPI` | 字典类型 + 字典数据 CRUD |
| 部门 | `DeptAPI` | 部门树形 CRUD |
| 文件 | `FileAPI` | 文件上传、MD5 秒传、下载 |
| 算法 | `AlgorithmAPI` | 算法 CRUD、版本管理、审核、导入导出 |
| 模型 | `ModelAPI` | 预测、评估、日志查询 |
| 数据集 | `DatasetAPI` | 数据集树形 CRUD |
| 数据项 | `DatasetItemAPI` | 数据项 CRUD、批量操作 |
| 图片文件 | `ItemFileAPI` | 数据项图片上传、编辑、删除 |
| 任务 | `TaskAPI` | 异步任务创建、状态查询、取消 |
| 图像历史 | `ImageInputHistoryAPI` | 图像输入历史记录 CRUD |

### 配置导出

```javascript
import { configJavaAxios, configPythonAxios } from 'dehaze-sdk-js';
```

### Axios 实例导出

```javascript
import { javaService, pythonService } from 'dehaze-sdk-js';
```

可用于 token 刷新后重发请求。

## 开发

### 构建项目

```bash
pnpm run build
```

这将在 `dist/` 目录下生成编译后的文件。

### 清理构建产物

```bash
pnpm run clean
```

### 运行测试

```bash
# 运行所有测试
pnpm run test

# 监听模式运行测试
pnpm run test:watch

# 生成测试覆盖率报告
pnpm run test:coverage

# 运行单个测试
npx vitest run test/user/user.test.ts --testNamePattern="正向测试：获取当前登录用户信息并验证数据完整性"
```

测试使用 Vitest 框架，需要后端服务运行在 `http://localhost:8989`。测试包含数据集、数据项、文件和导出任务的集成测试。

## 许可证

ISC
