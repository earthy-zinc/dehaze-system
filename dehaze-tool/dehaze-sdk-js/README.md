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
    config.baseURL = 'http://localhost:8080';
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
    config.baseURL = 'http://localhost:5000';
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

## API 列表

### 认证相关

- `AuthAPI.login` - 用户登录
- `AuthAPI.logout` - 用户登出
- `AuthAPI.getCaptcha` - 获取验证码

### 用户管理

- `UserAPI.getInfo` - 获取当前用户信息（昵称、头像、权限集合和角色集合）
- `UserAPI.getPage` - 获取用户分页列表
- `UserAPI.getFormData` - 获取用户表单详情
- `UserAPI.add` - 添加用户
- `UserAPI.update` - 修改用户
- `UserAPI.updatePassword` - 修改用户密码
- `UserAPI.deleteByIds` - 删除用户
- `UserAPI.downloadTemplate` - 下载用户导入模板
- `UserAPI.export` - 导出用户
- `UserAPI.import` - 导入用户

### 角色管理

- `RoleAPI.getPage` - 获取角色分页数据
- `RoleAPI.getOptions` - 获取角色下拉数据源
- `RoleAPI.getRoleMenuIds` - 获取角色的菜单ID集合
- `RoleAPI.updateRoleMenus` - 分配菜单权限给角色
- `RoleAPI.getFormData` - 获取角色表单数据
- `RoleAPI.add` - 添加角色
- `RoleAPI.update` - 更新角色
- `RoleAPI.deleteByIds` - 批量删除角色

### 菜单管理

- `MenuAPI.getRoutes` - 获取路由列表
- `MenuAPI.getList` - 获取菜单树形列表
- `MenuAPI.getOptions` - 获取菜单下拉数据源
- `MenuAPI.getFormData` - 获取菜单表单数据
- `MenuAPI.add` - 添加菜单
- `MenuAPI.update` - 修改菜单
- `MenuAPI.deleteById` - 删除菜单

### 字典管理

- `DictAPI.getDictTypePage` - 字典类型分页列表
- `DictAPI.getDictTypeForm` - 字典类型表单数据
- `DictAPI.addDictType` - 新增字典类型
- `DictAPI.updateDictType` - 修改字典类型
- `DictAPI.deleteDictTypes` - 删除字典类型
- `DictAPI.getDictOptions` - 获取字典类型的数据项
- `DictAPI.getDictPage` - 字典分页列表
- `DictAPI.getDictFormData` - 获取字典表单数据
- `DictAPI.addDict` - 新增字典
- `DictAPI.updateDict` - 修改字典项
- `DictAPI.deleteDictByIds` - 删除字典

### 部门管理

- `DeptAPI.getList` - 部门树形表格
- `DeptAPI.getOptions` - 部门下拉列表
- `DeptAPI.getFormData` - 获取部门详情
- `DeptAPI.add` - 新增部门
- `DeptAPI.update` - 修改部门
- `DeptAPI.deleteByIds` - 删除部门

### 文件管理

- `FileAPI.uploadCheck` - 文件上传检查
- `FileAPI.upload` - 上传文件
- `FileAPI.deleteByPath` - 删除文件

### 算法管理

- `AlgorithmAPI.getList` - 算法树形表格
- `AlgorithmAPI.getOption` - 获取模型下拉选项列表
- `AlgorithmAPI.getAlgorithmInfoById` - 获取算法详情
- `AlgorithmAPI.add` - 新增算法
- `AlgorithmAPI.update` - 修改算法
- `AlgorithmAPI.deleteByIds` - 删除算法

### 模型管理

- `ModelAPI.prediction` - 模型预测
- `ModelAPI.evaluation` - 模型评估

### 数据集管理

- `DatasetAPI.getList` - 数据集树形表格
- `DatasetAPI.getOptions` - 获取数据集下拉列表
- `DatasetAPI.getDatasetInfoById` - 根据Id获取数据集信息
- `DatasetAPI.getImageItem` - 获取数据集详细图片
- `DatasetAPI.add` - 新增数据集
- `DatasetAPI.update` - 修改数据集
- `DatasetAPI.deleteByIds` - 删除数据集
- `DatasetAPI.addDatasetItem` - 新增数据项
- `DatasetAPI.updateDatasetItem` - 更新数据项
- `DatasetAPI.deleteDatasetItem` - 删除数据项
- `DatasetAPI.uploadItemImage` - 上传数据项图片
- `DatasetAPI.updateItemImage` - 更新数据项图片
- `DatasetAPI.deleteItemImage` - 删除数据项图片

## 枚举类型

SDK 提供了常用的枚举类型供使用：

- `ResultEnum` - 响应码枚举

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

## 许可证

ISC
