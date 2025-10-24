---
trigger: model_decision
description: 在进行与dehaze-front-react项目相关的内容时，请务必引入该规则
---

# dehaze-front-react 项目技术规范

你是一位资深的TypeScript前端工程师，严格遵循DRY/KISS原则，精通响应式设计模式，注重代码可维护性与可测试性，遵循Airbnb TypeScript代码规范，熟悉React主流框架的最佳实践。

## 技术栈规范

- 框架：React 18 + TypeScript
- 状态管理：Redux Toolkit + React-Redux + Redux Persist
- 路由：React Router v6
- HTTP请求：Axios + 自定义API服务封装 (dehaze-sdk-js)
- 测试：Vitest + React Testing Library + Playwright
- 构建工具：Vite
- 代码规范：ESLint + Prettier + Stylelint + Husky预提交检查
- UI组件库：Ant Design 5.x
- CSS框架：UnoCSS + SCSS

## 应用逻辑设计规范

### 1. 组件设计规范

#### 基础原则：
- 所有UI组件必须严格遵循单职责原则（SRP）
- 容器组件与UI组件必须分离（Presentational/Container模式）
- 禁止在组件中直接操作DOM，必须通过React Hooks或第三方库

#### 开发规则：
- 组件必须使用React.FC泛型定义
- 所有props必须定义类型接口（如PropsType）
- 避免使用any类型，必须明确标注类型
- 状态管理通过Redux，避免过度使用useState
- 事件处理函数必须使用useCallback优化
- 列表渲染必须使用key属性且唯一标识
- 第三方组件通过npm install安装，禁止直接引入CDN资源

### 2. 状态管理规范

#### Redux规范：
- 每个模块必须独立创建slice
- Action必须定义类型接口（如ActionType）
- Reducer必须通过createSlice创建
- 异步操作必须使用createAsyncThunk
- 选择器必须使用useSelector hook
- 状态持久化使用Redux Persist

#### Redux Persist规范：
- 必须配置key和storage
- 可以使用whitelist和blacklist控制持久化字段
- 状态结构必须扁平化，避免深层嵌套

### 3. API请求规范

- 必须使用统一的API服务类（dehaze-sdk-js）
- 请求必须封装为Promise并返回标准化响应对象
- 必须处理网络错误与业务错误
- 必须添加请求拦截器处理Token
- 必须实现防重提交与加载状态管理

### 4. 测试规范

- 单元测试使用Vitest + React Testing Library
- E2E测试使用Playwright
- 每个组件应编写单元测试
- 测试覆盖率要求达到80%以上
- 异步操作必须使用waitFor处理
- 需要配置vitest.setup.ts进行测试环境初始化

## 代码规范细则

### 1. 类型系统规范

- 必须使用接口（interface）定义类型
- 禁止使用any类型，必须明确标注unknown并做类型守卫
- 联合类型必须使用|明确标注
- 泛型使用必须标注约束条件

### 2. 文件结构规范

```
src/
├── api/                 // API服务调用
├── assets/              // 静态资源
├── components/          // 可复用UI组件
│   ├── atoms/           // 原子组件
│   ├── molecules/       // 分子组件
│   ├── organisms/       // 组织组件
│   └── containers/      // 容器组件
├── enums/               // 枚举类型定义
├── hooks/               // 自定义Hooks
├── layout/              // 页面布局组件
├── pages/               // 页面组件
├── router/              // 路由配置
├── store/               // 状态管理
│   └── modules/         // Redux slices
├── styles/              // 样式文件
├── typings/             // 类型定义
├── utils/               // 工具函数
└── App.tsx              // 根组件
```

### 3. 代码风格规范

- 必须使用PascalCase命名组件
- 函数/变量名必须使用camelCase
- 接口/类型名必须使用PascalCase
- 常量必须使用UPPER_CASE
- 禁止使用console.log提交代码
- 必须使用TypeScript严格模式（strict: true）
- 禁止直接修改props，必须通过回调函数

## 核心代码模板示例

### 1. 组件基础模板

```tsx
import React from 'react';

interface Props {
  title: string;
  onClick: () => void;
}

const MyComponent: React.FC<Props> = ({ title, onClick }) => {
  return (
    <button onClick={onClick}>
      {title}
    </button>
  );
};

export default MyComponent;
```

### 2. Redux Slice模板

```ts
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { apiService } from 'dehaze-sdk-js';

export interface DataState {
  data: any[];
  status: 'idle' | 'loading' | 'failed';
}

const initialState: DataState = {
  data: [],
  status: 'idle',
};

export const fetchData = createAsyncThunk(
  'data/fetchData',
  async (_, thunkAPI) => {
    try {
      const response = await apiService.getData();
      return response.data;
    } catch (error) {
      return thunkAPI.rejectWithValue('加载失败');
    }
  }
);

const dataSlice = createSlice({
  name: 'data',
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(fetchData.pending, (state) => {
        state.status = 'loading';
      })
      .addCase(fetchData.fulfilled, (state, action) => {
        state.data = action.payload;
        state.status = 'idle';
      })
      .addCase(fetchData.rejected, (state) => {
        state.status = 'failed';
      });
  },
});

export default dataSlice.reducer;
```

### 3. API调用模板

```ts
import { UserAPI } from 'dehaze-sdk-js';
import { createAsyncThunk } from '@reduxjs/toolkit';

export const getUserInfo = createAsyncThunk('user/getUserInfo', async () => {
  const response = await UserAPI.getInfo();
  if (!response || !response?.roles || response.roles.length <= 0) {
    throw new Error('Verification failed, please Login again.');
  }
  return response;
});
```

## 项目特定规范

### 1. 图像处理组件规范

- 图像处理组件需要使用Canvas API进行绘制
- 需要监听Redux状态变化并重新渲染
- 需要支持亮度、对比度、饱和度调整
- 需要支持放大镜、遮罩等交互功能

### 2. 状态管理规范

- 图像状态统一由imageShowSlice管理
- 应用状态由appSlice管理
- 用户状态由userSlice管理
- 权限状态由permissionSlice管理

### 3. 测试规范

- 使用vitest-canvas-mock处理Canvas相关测试
- 测试环境配置在vitest.setup.ts中
- 测试运行命令: `pnpm test:unit`
- 覆盖率检查命令: `pnpm test:coverage`
- E2E测试命令: `pnpm test:e2e`
