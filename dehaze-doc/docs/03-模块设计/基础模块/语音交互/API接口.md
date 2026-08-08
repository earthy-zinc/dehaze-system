# 语音交互模块 API 接口

## 1. 文档概述

本文档定义 **语音交互** 模块的 HTTP API 和 WebSocket 规范。

- **基础路径**：`/api/v1/voice`
- **WebSocket**：`ws://{host}/ws/asr`（流式语音识别）
- **公共约定**：参见 [02-系统架构/04-API规范.md](../../../02-系统架构/04-API规范.md)

## 2. 接口清单

### 2.1 ASR 识别接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/asr/stream-session` | POST | 创建流式 ASR 会话（获取 WebSocket 连接信息） | - | F-VS-001 |
| `/api/v1/voice/asr/offline` | POST | 离线 ASR 识别（提交完整音频文件） | - | F-VS-001 |
| `/api/v1/voice/asr/result/{sessionId}` | GET | 查询流式 ASR 会话的最终识别结果 | - | F-VS-001 |

**流式 ASR 流程**：
```
1. POST /asr/stream-session → 获取 sessionId + wsUrl
2. 前端通过 wsUrl 建立 WebSocket → 推送音频块 → 接收增量/最终文本
3. 前端停止录音后 → GET /asr/result/{sessionId} 获取最终完整文本
```

### 2.2 TTS 合成接口

| 路径 | 方法 | 功能描述 | 权限标识 | 关联功能点 |
|------|------|---------|---------|-----------|
| `/api/v1/voice/tts` | POST | 文本转语音（返回音频 URL，支持流式） | - | F-VS-002 |
| `/api/v1/voice/tts/voices` | GET | 可用音色列表 | - | F-VS-002 |

**TTS 请求参数**：

```json
{
  "text": "处理完成，PSNR 28.5，SSIM 0.92",
  "voiceId": "aixia",
  "speed": 1.0,
  "stream": true
}
```

## 3. 权限标识

语音交互模块为系统基础能力，登录用户均可使用，无需特殊权限标识。
