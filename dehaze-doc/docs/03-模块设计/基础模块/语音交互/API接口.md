# 语音交互模块 API 接口

## 1. 文档概述

本文档定义 **语音交互** 模块的 HTTP API 和 WebSocket 规范。

- **基础路径**：`/api/v1/voice`
- **流式协议**：WebSocket（ASR 实时语音识别，端点 `ws://{host}/ws/asr`）+ HTTP 流式响应（TTS 语音合成）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

## 2. 接口清单

### 2.1 ASR 识别接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/asr/stream-session` | POST | 创建流式 ASR 会话（获取 WebSocket 连接信息） | - | F-VS-001 |
| `/api/v1/voice/asr/offline` | POST | 离线 ASR 识别（提交完整音频文件） | - | F-VS-001 |
| `/api/v1/voice/asr/result/{sessionId}` | GET | 查询流式 ASR 会话的最终识别结果 | - | F-VS-001 |

### 2.2 TTS 合成接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/tts` | POST | 文本转语音（返回音频 URL 或 HTTP 流式响应） | - | F-VS-002 |
| `/api/v1/voice/tts/voices` | GET | 可用音色列表 | - | F-VS-002 |

### 2.3 热词管理接口

> 热词用于提升 ASR 对领域专业术语的识别率。全局热词（所有用户生效）由管理员配置，用户级热词（仅本人生效）由用户自行管理。

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/hotwords` | GET | 查询当前用户热词列表 | - | F-VS-004 |
| `/api/v1/voice/hotwords` | POST | 新增用户热词 | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/{id}` | DELETE | 删除用户热词 | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/global` | GET | 查询全局热词列表 | - | F-VS-004 |
| `/api/v1/voice/hotwords/global` | POST | 新增全局热词（管理员） | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/global/{id}` | DELETE | 删除全局热词（管理员） | `voice:hotword:edit` | F-VS-004 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 语音交互模块基础能力（ASR/TTS/热词查询），登录用户均可使用 |
| `voice:hotword:edit` | 热词新增/删除（用户级仅本人，全局需管理员） |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0230` | token 无效或已过期 | 未登录访问 |
| `A0401` | 请求资源不存在 | ASR 会话不存在、音色 ID 不存在 |
| `A0500` | 业务异常 | ASR 并发会话超上限、录音时长超限、AI 积分余额不足 |
| `A0600` | 操作失败 | FunASR 调用失败、云 TTS API 不可用、流式合成超时 |
