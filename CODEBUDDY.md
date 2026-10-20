# CODEBUDDY.md

本文件提供每会话必需的基础口径：仓库定位、生命周期纪律、读代码看不出的关键约定、文档入口。设计与契约的权威源是 `clarity-doc/docs/`；通用行为红线见 `.codebuddy/rules/`。

## 仓库概览

`clarity-vision` 是澄莹智能图像处理平台（Clarity Vision）的单仓库多端 monorepo：一套 API 契约由三端后端（Java / Go / Python）共同实现，配多套前端、JS / Android SDK、算法与推理服务、VitePress 设计文档。

`clarity-python` 是 AI 域的行为事实源。AI 域按端点分 A / B 两类：A 类三端各自原生实现同一套契约，改动须三端同步；B 类仅 Python 实现，java / go 按各自转发白名单转发到 Python(8991)。具体划分、白名单与易错细则见 skill `project-knowledge`；接口行为存疑时以 Python 实现 + `clarity-doc` 为准，不要凭直觉在三端各写一套。

## 跨端同步

改动涉及 API 请求/响应参数增删改、数据流或架构设计时，下面这些位置一并检查与修改，不能只改一处：

- 三端后端 clarity-java / clarity-python / clarity-go 及其单元测试
- SDK 层 clarity-sdk-js（`test/` 是 SDK 接口测试）与 clarity-android/sdk
- 用到该 API 的各前端与客户端：clarity-front-vue、clarity-front-react、clarity-desktop-web、clarity-mobile、clarity-paper、clarity-taro、clarity-uniapp、clarity_flutter、clarity-react-native、clarity-android
- `clarity-doc` 对应文档

新增会消费该 API 的仓库时，同步补进这份清单。

## 生命周期

三端后端（含 MCP 网关与 `clarity-service/*` 推理服务）统一由根目录 `scripts/run.py` 管理启动 / 停止 / 重启 / 查看状态 / 查看日志，不要在子项目目录里手动起停，更不要私自重启或停止服务；需要重启先告知用户。日志统一在各服务 `logs/{yyyy-MM-dd}/` 下，启动指引见根目录 `README.md`。

服务与端口清单、`run.py` 子命令、启动与部署与本地环境搭建步骤见 `clarity-doc/docs/02-系统架构/06-部署架构.md`；日志路径与切割规则见 `clarity-doc/docs/02-系统架构/07-日志架构设计.md`。

## 关键约定

- 数据层：三端共享 `config/sql/schema/*.sql` 与 `data/*.sql`（`sys_*` 种子）；存储后端走策略模式适配（local / MinIO / 阿里云 OSS / nginx-static），URL 不落库，由 `baseUrl + object_name` 运行时拼接。
- 配置：所有连接参数走根 `.env`。

## 环境

dev 与 pre 两个环境。构建在 dev 做，部署与验证在 pre；pre 是"只测不开发"的共享环境，不改被 git 跟踪的文件——改动一律本地改 → `git push origin master` → pre `git pull --ff-only`。连接用 `ssh earthy-dev` / `ssh earthy-pre`（别名自带端口，勿用 IP 直连）；Redis 排障 dev 用 db1、pre 用 db2；页面测试与验证的证据落 `logs/test-evidence/{yyyy-MM-dd}/`（`logs/` 已被 gitignore）。联调、K8s 排障、镜像构建与浏览器调试的操作细节见 skill `project-knowledge` 的 `references/06-服务器与联调环境.md`。

## 命令落点

- 测试、lint 与静态检查门禁见 `clarity-doc/docs/02-系统架构/05-测试架构.md`。
- 轮末收口 hook：`.codebuddy/settings.json` 注册 PostToolUse / Stop / SessionEnd，`Stop` 批量格式化 + 静态检查。它静默、失败不阻断、不是门禁，不要为格式化或检查再单独跑一轮命令；设计与排障见 `clarity-doc/docs/02-系统架构/05-测试架构.md`。

## 先读什么

- 设计权威源：`clarity-doc/docs/`（总体架构 / 数据库 / API / 测试 / 部署 / 模块设计）。
- 行为规范：`.codebuddy/rules/`——`01-common-rules`（沟通表达、真实性与不造假、验证与举证，始终生效）、`02-coding-rules`（编码与注释，编辑代码文件时生效）、`03-writing-rules`（写作规范，始终生效）。
- 程序性知识与流程：skill `project-knowledge`（动手写代码、评审、排查、做计划前调用）、`doc-writer`（编写与整理项目文档）、`testing`（写与跑测试、静态门禁）。
- 冲突优先级：`rules` > skill 事实记录；与 `clarity-doc` 或代码冲突时以磁盘代码与实测为准。
