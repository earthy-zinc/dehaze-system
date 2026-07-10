# 前端开发规范

适用于：`dehaze-front-vue`、`dehaze-front-react` 及相关前端项目

## API 调用

- 统一通过 `dehaze-sdk-js` 库调用后端接口
- 必须区分：网络错误 vs 业务错误，并在 UI 层有可见反馈

## 工程化

- ESLint/Prettier/Stylelint/Husky
- .env 管理环境变量，禁止在代码里硬编码敏感配置

## 目录结构与命名

- 目录结构遵循项目现状；新增模块保持一致
- 命名：
  - 组件 PascalCase
  - 变量/函数 camelCase
  - 常量 UPPER_CASE
- 禁止提交 `console.log`

## React 开发规范

### 组件设计与分层

- 组件必须单一职责（SRP）
- 展示组件与容器组件分离（避免业务逻辑散落到 UI 组件）
- 避免直接操作 DOM；优先用 React 机制（ref/hooks）或成熟库

### 状态管理（Redux Toolkit）

- 业务状态优先进入 Redux（尤其是跨页面、跨组件共享的状态）
- 局部 UI 状态允许用 `useState`（不要为了"禁止 useState"而滥用 Redux）
- 异步：优先 `createAsyncThunk`，并维护 loading/error 状态
- 持久化：只 whitelist 必要字段，避免污染与版本迁移成本

## Vue 开发规范

### 组件规范（Composition API）

- 默认使用 `<script setup lang="ts">`
- 组件命名 PascalCase，props 用类型定义
- 复杂逻辑抽 hooks（composables），避免组件过胖

### Pinia 规范

- store：`useXxxStore` 命名
- 跨模块依赖要克制，避免循环依赖
- 异步与副作用集中在 actions，UI 只调用 actions

## 前端测试规范

### 目录与命名

- 建议同级 `__tests__/`
- `*.spec.ts` / `*.test.ts`

### 测试原则

- Mock：
  - 网络请求：`vi.mock` 或 MSW（若项目引入）
  - 时间/随机数：必须可控
- Canvas：需要时用 `vitest-canvas-mock`
- 关注"行为"，不要依赖实现细节
- 覆盖率目标可设为 ≥80%，但以项目 vitest 配置为准
- 避免弱断言：断言关键渲染、关键交互、副作用触发
