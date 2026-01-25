---
# 注意不要修改本文头文件，如修改，CodeBuddy（内网版）将按照默认逻辑设置
type: always
---
# React 开发规范

## 1. 组件设计与分层

- 组件必须单一职责（SRP）
- 展示组件与容器组件分离（避免业务逻辑散落到 UI 组件）
- 避免直接操作 DOM；优先用 React 机制（ref/hooks）或成熟库

## 2. 状态管理（Redux Toolkit）

- 业务状态优先进入 Redux（尤其是跨页面、跨组件共享的状态）
- 局部 UI 状态允许用 `useState`（不要为了“禁止 useState”而滥用 Redux）
- 异步：优先 `createAsyncThunk`，并维护 loading/error 状态
- 持久化：只 whitelist 必要字段，避免污染与版本迁移成本