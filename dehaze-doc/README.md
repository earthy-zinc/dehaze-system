# 土味锌的阅读笔记

[![VuePress](https://img.shields.io/badge/VuePress-2.x-brightgreen)](https://vuepress.vuejs.org/)
[![License](https://img.shields.io/github/license/earthy-zinc/reading-note)](LICENSE)

> 全栈开发学习笔记

这是一个全面的全栈开发学习笔记项目，涵盖了前端、后端、运维、算法等多个技术领域，旨在记录和分享软件开发的各类知识。

## 📚 内容概览

本项目包含了丰富的技术文档，覆盖以下主要领域：

- **前端开发**: HTML、CSS、JavaScript、Vue、React、TypeScript 等
- **后端开发**: Java、Python、Go、Spring、数据库、中间件等
- **学术课程**: 深度学习、论文阅读、研究日常等
- **项目文档**: 图像去雾系统、土味商城等实际项目文档
- **通用工具**: Git、PS、UML、快捷键等开发工具使用指南

## 🚀 快速开始

### 本地开发

```bash
# 安装依赖
pnpm install

# 启动开发服务器
pnpm run docs:dev
```

### 构建部署

```bash
# 构建静态文件
pnpm run docs:build
```

## 📂 项目结构

```
dehaze-doc/
├── docs/                  # 文档源文件
│   ├── .vuepress/         # VuePress 配置
│   ├── 前端开发/           # 前端技术文档
│   ├── 后端开发/           # 后端技术文档
│   ├── 学术课程/           # 学术相关文档
│   ├── 通用工具/           # 开发工具文档
│   ├── 项目文档/           # 项目相关文档
│   └── README.md          # 首页文档
├── package.json           # 项目配置文件
└── README.md              # 项目说明文件
```

## ⚙️ 技术栈

- [VuePress](https://vuepress.vuejs.org/) - 静态站点生成器
- [VuePress Theme Hope](https://theme-hope.vuejs.press/) - 主题框架
- [Vite](https://vitejs.dev/) - 构建工具
- [TypeScript](https://www.typescriptlang.org/) - 类型检查
- [Sass](https://sass-lang.com/) - CSS 扩展语言
- [Mermaid](https://mermaid-js.github.io/) - 图表绘制工具
- [ECharts](https://echarts.apache.org/) - 数据可视化库

## 🌐 部署方式

本文档支持多种部署方式：

1. **华为云DevCloud**: 通过流水线实现自动化部署
2. **静态文件部署**: 构建后可部署到任何静态文件服务器
3. **Gitee Pages**: 可配置自动部署到 Gitee Pages

## 📄 许可证

本项目采用 Apache License 2.0 许可证，详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **作者**: Earthy Zinc
- **仓库**: [https://gitee.com/earthy-zinc/reading-note](https://gitee.com/earthy-zinc/reading-note)