# CODEBUDDY.md

本文件为 CodeBuddy 提供**每会话必需的基础口径**（仓库定位、生命周期纪律、读代码看不出的关键约定与文档入口）。设计与契约的权威源是 `dehaze-doc/docs/`，本文件不重复其内容。

## 仓库概览

`dehaze-system` 是 **AI 图像处理平台**的单仓库多端 monorepo：一套 API 契约由三端后端（Java / Go / Python）共同实现，配多套前端、JS / Android SDK、算法与推理服务、VitePress 设计文档。

**`dehaze-python` 是 AI 域的行为事实源，但 AI 域并非"只存在于 Python"**：AI 域按端点分 A/B 两类——**A 类**（会话/消息 CRUD、模型/供应商/密钥、计费/账单/余额、偏好、可观测查询、评测中心）由 **java / go / python 各自原生实现同一套契约，改动须三端同步**；**B 类**（对话推理 SSE、agent/skill 试运行、知识库索引与检索、评测派发、MCP、语音 ASR/TTS、兼容协议入口）**仅 Python 实现，java / go 按白名单转发**到 Python(8991)。B 类清单以各端转发白名单为唯一口径（java `service/client/AiProxyRoutes.java`、go `internal/router/ai_proxy.go` + `voice_proxy.go`），且两端白名单并非逐条相同；接口行为存疑时以 Python 实现 + `dehaze-doc` 为准，不要凭直觉在三端各写一套。

### 生命周期（违反即事故）

三端后端（含 MCP 网关与 `dehaze-service/*` 推理服务）统一由根目录 `scripts/run.py` 管理，**切勿私自重启或停止服务**。

服务与端口清单、`run.py` 子命令、启动 / 部署 / 本地环境搭建步骤见 `dehaze-doc/docs/02-系统架构/06-部署架构.md`；日志路径与切割规则见 `dehaze-doc/docs/02-系统架构/07-日志架构设计.md`。

## 关键约定（读代码看不出的部分）

- **数据层**：三端共享 `config/sql/schema/*.sql` 与 `data/*.sql`（`sys_*` 种子）；存储后端走策略模式适配（local / MinIO / 阿里云 OSS / nginx-static），**URL 不落库**，由 `baseUrl + object_name` 运行时拼接。
- **配置**：所有连接参数走根 `.env`（`MYSQL_*` / `REDIS_*` / `MINIO_*` …）。

## 命令落点

- **测试 / lint / 静态检查门禁**：`dehaze-doc/docs/02-系统架构/05-测试架构.md`；任何 Python 代码改动交付前必须**终端**跑通 `bash dehaze-python/scripts/check.sh`（唯一判定入口，IDE 诊断不作依据）。
- **轮末收口 hook**：`.codebuddy/settings.json` 注册 PostToolUse / Stop / SessionEnd，`Stop` 批量格式化 + 静态检查。它**静默、失败不阻断、不是门禁**，**不要**为格式化或检查再单独跑一轮命令；设计与排障见 `dehaze-doc/docs/02-系统架构/05-测试架构.md`。

## 先读什么

- **设计权威源**：`dehaze-doc/docs/`（总体架构 / 数据库 / API / 测试 / 部署 / 模块设计）。
- **行为规范**：`.codebuddy/rules/`（`01-common-rules` 为 always-applied，含编码 / 注释 / 跨端同步清单；核心红线：**禁止兼容历史烂逻辑**）。
- **程序性知识**：skill `dehaze-project-knowledge` —— 动手写代码 / 评审 / 排查 / 做计划**前先调用**（架构禁区、跨端契约口径、数据配置、测试基建、语言栈与环境坑）。
- 冲突优先级：`rules` > skill 事实记录；与 `dehaze-doc` / 代码冲突时以磁盘代码与实测为准。
