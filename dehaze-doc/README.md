# 图像去雾系统 - 项目技术文档

[![VitePress](https://img.shields.io/badge/VitePress-1.x-brightgreen)](https://vitepress.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

> 本文档是 DehazeSystem 项目的**唯一权威文档来源**，涵盖产品设计、系统架构、模块设计等完整项目文档。

## 📚 文档结构

```
docs/
├── 01-产品设计/          # 产品概述、UI/UX 设计规范
├── 02-系统架构/          # 总体架构、数据库、API 规范、部署架构等
├── 03-模块设计/          # 基础模块 + 核心模块的详细设计
│   ├── 基础模块/         # 认证、用户、角色、部门、菜单、字典、文件、任务
│   └── 核心模块/         # 图像输入、去雾处理、效果对比、数据集、算法管理、算法选择
└── 04-改造计划/          # 基础设施问题记录与改造规划
```

## 🚀 快速开始

```bash
# 安装依赖
pnpm install

# 启动开发服务器
pnpm run docs:dev

# 构建静态文件
pnpm run docs:build

# 预览构建结果
pnpm run docs:preview
```
