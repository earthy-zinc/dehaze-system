---
trigger: model_decision
description: 在进行与dehaze-front-vue项目相关的操作时请阅读该规则
---

## 项目技术栈规范
- **核心框架**：Vue 3 + Vite + TypeScript
- **UI组件库**：Element Plus
- **状态管理**：Pinia
- **路由管理**：Vue Router
- **构建工具**：Vite
- **包管理器**：pnpm
- **代码规范**：ESLint + Prettier + Stylelint
- **测试工具**：Vitest + Playwright
- **CSS框架**：UnoCSS

## Vue 3 Composition API 组件开发规范
- 使用 `<script setup lang="ts">` 语法糖
- 组件命名采用 PascalCase
- 使用 TypeScript 接口明确定义 Props 类型
- 使用 Composition API 生命周期钩子 (onMounted、onUnmounted 等)

## Pinia 状态管理规范
- 使用 defineStore 创建 store
- Store 命名采用 useXxxStore 格式
- 避免直接修改 state，应通过 actions 方法
- Store 模块化组织，按业务功能划分

## API 调用规范
- 使用 dehaze-sdk-js 封装的 API 模块
- Java 后端 API 通过 UserAPI, RoleAPI 等模块调用
- Python 后端 API 通过 AlgorithmAPI, ModelAPI 等模块调用
- 统一通过 [src/utils/request.ts](../../dehaze-front-vue/src/utils/request.ts) 配置拦截器

## TypeScript 类型定义规范
- 使用 TypeScript 接口/类型别名定义数据结构
- 避免使用 any 类型，强制类型推断
- 全局类型定义放在 src/typings 目录
- 组件 Props 使用 defineProps<T>() 定义

## 测试规范
- **单元测试**：使用 Vitest，测试覆盖率要求 ≥ 80%
- **端到端测试**：使用 Playwright
- 测试文件命名为 *.test.ts 或 *.spec.ts
- 使用 vitest-canvas-mock 处理 canvas 相关测试

## 代码规范和工程化
- 遵循 ESLint + Prettier 统一代码格式
- 使用 UnoCSS 原子化 CSS
- 使用 SCSS 预处理器
- 通过 .env 文件管理环境变量

## Git 版本控制规范
- 遵循 Conventional Commits 标准
- 使用 commitizen 标准化提交信息
- 主分支为 main，开发分支为 feature/xxx 或 bugfix/xxx

## 安全规范
- 对用户输入进行验证和清理
- 使用 Token 进行用户认证
- 敏感数据进行加密传输和存储
- 使用 HTTPS 进行所有网络通信
