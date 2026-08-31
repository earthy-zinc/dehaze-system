# 语音交互模块 API 接口

## 1. 文档概述

本文档定义 **语音交互** 模块的 HTTP API 和 WebSocket 规范。

- **基础路径**：`/api/v1/voice`
- **流式协议**：WebSocket（ASR 实时语音识别，端点 `ws://{host}/ws/asr`）+ HTTP 流式响应（TTS 语音合成）

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
| `/api/v1/voice/tts/audio/{cacheKey}` | GET | 缓存音频下载（AES 解密后流式返回，仅本人缓存可访问） | - | F-VS-002 |

### 2.3 热词管理接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/hotwords` | GET | 查询当前用户热词列表 | - | F-VS-004 |
| `/api/v1/voice/hotwords` | POST | 新增用户热词 | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/{id}` | DELETE | 删除用户热词 | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/global` | GET | 查询全局热词列表 | - | F-VS-004 |
| `/api/v1/voice/hotwords/global` | POST | 新增全局热词（管理员） | `voice:hotword:edit` | F-VS-004 |
| `/api/v1/voice/hotwords/global/{id}` | DELETE | 删除全局热词（管理员） | `voice:hotword:edit` | F-VS-004 |

### 2.4 服务状态接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/service/status` | GET | 查询 ASR/TTS 服务运行状态（引擎在线状态/并发会话数/模型加载状态，管理端服务监控） | `voice:service:monitor` | F-VS-001/002 |

### 2.5 语音引擎管理接口（管理端）

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/providers` | GET | 引擎分页列表（含健康状态看板） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers` | POST | 新增引擎（provider_code 唯一，删除后不可复用） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}` | PUT | 更新引擎（含 `is_default`/`status`） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}` | DELETE | 删除引擎（有启用模型引用时需先禁用，provider_code 保留） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/enabled` | GET | 指定能力维度启用引擎列表（`engine_type=asr\|tts`） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}/test-connection` | POST | 连通性测试（结果仅提示不阻断保存） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}/keys` | GET | 查询引擎 API Key 列表 | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}/keys` | POST | 新增 API Key（加密存储，返回 Key ID 不返回明文） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}/keys/{keyId}` | PUT | 更新 API Key（禁用/权重/限额等） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/providers/{id}/keys/{keyId}` | DELETE | 删除 API Key（物理删除，状态控制） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/models` | GET | 模型/音色列表（按 `engine_type` 筛选） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/models` | POST | 新增模型/音色（`model_id` 删除后不可复用） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/models/{modelId}` | PUT | 更新模型/音色（含 params） | `voice:engine:manage` | F-VS-005 |
| `/api/v1/voice/models/{modelId}` | DELETE | 删除模型/音色（`model_id` 保留） | `voice:engine:manage` | F-VS-005 |

## 3. 权限标识汇总

| 权限标识 | 说明 |
|---------|------|
| - | 语音交互模块基础能力（ASR/TTS/热词查询），登录用户均可使用 |
| `voice:hotword:edit` | 热词新增/删除（用户级仅本人，全局需管理员） |
| `voice:service:monitor` | 语音服务状态监控（管理端，仅管理员） |
| `voice:engine:manage` | 语音引擎管理（注册表/API Key/模型音色配置，管理端，仅管理员） |

## 4. 业务错误码

| 错误码 | 说明 | 触发场景 |
|--------|------|---------|
| `A0230` | token 无效或已过期 | 未登录访问 |
| `A0400` | 请求参数错误 | TTS 空/超长文本、不支持的音色、离线 ASR 非 WAV/PCM 格式或空文件、热词内容为空 |
| `A0401` | 请求资源不存在 | ASR 会话不存在、热词不存在、缓存音频不存在或已过期 |
| `A0301` | 权限不足 | 普通用户管理全局热词、访问他人 ASR 会话/热词 |
| `A0500` | 业务异常 | ASR 并发会话超上限、热词数量超上限、本地 TTS 引擎失败、纯云端部署下默认引擎冲突（`is_default` 指向 `local` 而本地引擎不可用，抛错不降级） |
| `A0682` | 配额不足或欠费熔断 | ASR/TTS 调用前 AI 积分余额/配额校验不通过 |
| `C0001` | 第三方服务调用失败 | 云端 ASR/TTS 调用失败 |
