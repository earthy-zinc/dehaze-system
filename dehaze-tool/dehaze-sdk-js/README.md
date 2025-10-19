# Dehaze SDK for JavaScript

JavaScript SDK for Dehaze System API

## 安装

```bash
npm install dehaze-sdk-js
```

## 构建

```bash
npm run build
```

## 使用方法

```javascript
import { UserAPI } from 'dehaze-sdk-js';

UserAPI.getInfo().then(res => {
  console.log(res);
});
```

## 测试

项目包含一个测试目录，用于验证SDK的功能。

### 简单测试

进入测试目录并安装依赖：

```bash
cd test
npm install
```

### TypeScript 兼容性测试

SDK 的 TypeScript 兼容性测试使用 TypeScript 编写的测试项目来验证 SDK 在 TypeScript 环境中的类型定义和功能是否正常。

1. 构建测试项目：
   ```bash
   cd test
   npm run build
   ```

2. 运行 TypeScript 测试：
   ```bash
   npm test
   ```

### 完整功能测试

完整功能测试需要启动一个模拟服务器来响应SDK的API调用。

1. 首先在一个终端窗口启动模拟服务器：

```bash
cd test
npm run build
npm run test-server
```

或者使用 ts-node 直接运行（无需预先构建）：

```bash
cd test
npm run dev-server
```

2. 在另一个终端窗口运行测试：

```bash
cd test
npm test
```

或者使用 ts-node 直接运行（无需预先构建）：

```bash
cd test
npm run dev
```

测试将执行以下操作：
- 获取当前用户信息
- 获取用户分页列表
- 获取用户表单详情
- 添加新用户
- 修改用户信息
- 修改用户密码
- 删除用户

## 许可证

ISC

# Dehaze JavaScript SDK

Dehaze 系统的 JavaScript SDK，用于简化前端项目与后端 API 的交互。

## 安装

```bash
pnpm add dehaze-sdk-js
pnpm add file:../dehaze-tool/dehaze-sdk-js
```

## 基础用法

默认情况下，SDK 会使用默认开发环境的配置：

```javascript
import { getUserList } from 'dehaze-sdk-js';

const users = await getUserList();
```

## API 列表

### 用户管理

- `getUserList` - 获取用户列表
- `getUserInfo` - 获取用户信息
- `createUser` - 创建用户
- `updateUser` - 更新用户
- `deleteUser` - 删除用户

### 角色管理

- `getRoleList` - 获取角色列表
- `getRoleInfo` - 获取角色信息
- `createRole` - 创建角色
- `updateRole` - 更新角色
- `deleteRole` - 删除角色

### 菜单管理

- `getMenuList` - 获取菜单列表
- `getMenuInfo` - 获取菜单信息
- `createMenu` - 创建菜单
- `updateMenu` - 更新菜单
- `deleteMenu` - 删除菜单

### 字典管理

- `getDictList` - 获取字典列表
- `getDictInfo` - 获取字典信息
- `createDict` - 创建字典
- `updateDict` - 更新字典
- `deleteDict` - 删除字典

### 部门管理

- `getDeptList` - 获取部门列表
- `getDeptInfo` - 获取部门信息
- `createDept` - 创建部门
- `updateDept` - 更新部门
- `deleteDept` - 删除部门

### 文件管理

- `uploadFile` - 上传文件
- `downloadFile` - 下载文件
- `deleteFile` - 删除文件

### 算法管理

- `getAlgorithmList` - 获取算法列表
- `getAlgorithmInfo` - 获取算法信息
- `createAlgorithm` - 创建算法
- `updateAlgorithm` - 更新算法
- `deleteAlgorithm` - 删除算法

### 模型管理

- `getModelList` - 获取模型列表
- `getModelInfo` - 获取模型信息
- `createModel` - 创建模型
- `updateModel` - 更新模型
- `deleteModel` - 删除模型

### 数据集管理

- `getDatasetList` - 获取数据集列表
- `getDatasetInfo` - 获取数据集信息
- `createDataset` - 创建数据集
- `updateDataset` - 更新数据集
- `deleteDataset` - 删除数据集
