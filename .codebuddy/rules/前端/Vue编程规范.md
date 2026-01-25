---
# 注意不要修改本文头文件，如修改，CodeBuddy（内网版）将按照默认逻辑设置
type: always
---
# Vue 开发规范

## 1. 组件规范（Composition API）

- 默认使用 `<script setup lang="ts">`
- 组件命名 PascalCase，props 用类型定义
- 复杂逻辑抽 hooks（composables），避免组件过胖

## 2. Pinia 规范

- store：`useXxxStore` 命名
- 跨模块依赖要克制，避免循环依赖
- 异步与副作用集中在 actions，UI 只调用 actions