# dehaze-voice（语音推理引擎服务，8993）

`dehaze-python` 的 ASR / TTS 推理能力外置形态（边界见 [AI 能力链路协作设计](../../dehaze-doc/docs/02-系统架构/09-AI能力链路协作设计.md)）。

**只做"引擎参数进、识别/合成结果出"**，服务侧不承担业务语义：

| 不承担                                                       | 归属                                                      |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| MySQL / Redis / ES / MinIO 连接                              | 无（核心不变式：零业务状态，可无状态重启与水平扩容）      |
| 用户鉴权、并发上限、计费、会话态、热词存储                   | `dehaze-python` 编排侧（Redis）                           |
| 模型/音色目录（`sys_voice_model`）解析                       | 编排侧解析后随请求下发                                    |
| ASR/VAD 空闲与时长超时（空闲 60s / 最长 60s / 下行收尾 10s） | 编排侧（前端契约）；本服务仅保留"无输入 30s 自动收尾"兜底 |

## 快速开始

```bash
cd dehaze-service/dehaze-voice
uv sync                        # 首次安装引擎依赖（含 torch/torchaudio，耗时较长）
.venv/bin/python -m app.main   # 默认 0.0.0.0:8993，等价于 run.py start voice
```

`scripts/run.py` 统一管理生命周期（`run.py start|stop|restart|ps|logs voice`），日志落
`dehaze-service/dehaze-voice/logs/{yyyy-MM-dd}/`（`info.log` / `error.log` 为 NDJSON，
`fields.service=voice`；`console.log` 由 run.py 重定向）。

单进程启动（`workers=1`）：引擎模型每进程一份，多实例扩容靠起多个服务实例，而非 uvicorn 多 worker。
`uvicorn` 自身日志压到 warning 且关闭 access log（ACCESS 日志由应用中间件统一记录）。

容器镜像（`python:3.11-slim` + uv，模型经 `/models` 持久卷；CUDA 运行库随 torch wheel 装入，无需 NVIDIA 基座）：

```bash
docker build -t dehaze-voice -f dehaze-service/dehaze-voice/Dockerfile dehaze-service/dehaze-voice
```

```bash
# 冒烟：探针与指标同样校验 Bearer
curl -H "Authorization: Bearer ${INTERNAL_SERVICE_KEY:-local-no-key}" http://127.0.0.1:8993/ready
```

## 端点

**全部端点**（含探针与 `/metrics`）均校验 `Authorization: Bearer ${INTERNAL_SERVICE_KEY}`
（全局 ASGI `BearerAuthMiddleware` + `hmac.compare_digest`），未携带或密钥错误返回 401；
`INTERNAL_SERVICE_KEY` 未配置（空串）时一律 401（fail closed，不会退化为无鉴权）。
内网服务不开放 Swagger/OpenAPI 文档面（`docs_url` / `redoc_url` / `openapi_url` 均为 `None`）。

| 端点              | 方法              | 说明                                                                                                                                  |
| ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `/v1/asr/offline` | POST（multipart） | `audio`（WAV/PCM 16kHz/16bit/mono）、`model_id`（ModelScope ID）、`hotwords`（JSON 数组字符串，可空）→ `{"text"}`                     |
| `/v1/asr/stream`  | WS                | 握手头携带 Bearer，query 仅 `model_id`；见下方流式协议                                                                                |
| `/v1/tts`         | POST（JSON）      | `text`、`voice_id`、`voice_params`、`speed`（0.5~2.0）、`format`（mp3/wav/pcm）、`sample_rate` → 音频字节（`Content-Type` 按 format） |
| `/v1/models`      | GET               | 能力发现（§3.6）：引擎、已加载/已缓存模型与音色、可下载项、格式与采样率                                                               |
| `/v1/status`      | GET               | `{"asr":{"loaded","model_id","device"},"tts":{"loaded","voice_id","device"}}`                                                         |
| `/health`         | GET               | liveness：进程存活即 200，不触发任何模型加载/下载                                                                                     |
| `/ready`          | GET               | readiness：引擎依赖可加载即 200，否则 503（`NOT_READY`）                                                                              |
| `/metrics`        | GET               | Prometheus 指标（`dehaze_http_requests_total`、`dehaze_http_request_duration_seconds`、`dehaze_inference_duration_seconds`）          |

错误统一为 `{"code","message"}` + HTTP 状态语义（400 参数非法 / 401 未授权 / 422 音频非法 /
500 推理失败 / 503 模型不可用）；调用方（编排侧 `X-Trace-Id` 透传、Prometheus `authorization`）均携带同一密钥。

### 离线识别

```bash
curl -X POST http://127.0.0.1:8993/v1/asr/offline \
  -H "Authorization: Bearer local-no-key" \
  -F "audio=@sample.wav" \
  -F "model_id=iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
  -F 'hotwords=["去雾","图像增强"]'
```

### 语音合成

```bash
curl -X POST http://127.0.0.1:8993/v1/tts \
  -H "Authorization: Bearer local-no-key" -H "Content-Type: application/json" \
  -d '{"text":"你好","voice_id":"zh_CN-huayan-medium",
       "voice_params":{"onnx":"zh_CN-huayan-medium.onnx","size":63201294,
                       "urls":["https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/{filename}"]},
       "speed":1.0,"format":"mp3","sample_rate":24000}' --output out.mp3
```

### 流式识别（WS 协议）

```
握手：GET /v1/asr/stream?model_id=<ModelScope ID>   Header: Authorization: Bearer <key>
  ├─ 首条控制消息（文本）：{"type":"config","hotwords":[...],"session_id":"<可选幂等键>"}
  ├─ 上行：二进制 PCM（16kHz/16bit/mono，建议 100~500ms/块）；结束发文本 "EOS"
  └─ 下行：{"text":"<当前累计文本>","is_final":false} … {"text":"…","is_final":true}
```

- 每累计 1 秒音频触发一次分段转写，下行按"当前累计文本"覆盖式给出。
- 热词按会话传入：引擎仅在本次会话内生效，进程内无全局热词表（多用户并发互不串扰）。
- `session_id` 幂等：同一实例内重复建连直接拒绝（`SESSION_CONFLICT`），且不进入识别。
- 关闭码：4400 参数非法（缺 `model_id` / config 非法）、4401 未授权（握手即拒绝）、
  4409 会话冲突、4500 推理失败、4503 模型不可用；正常收尾关闭码 1000。

## 环境变量

| 变量                                                                                                      | 默认                       | 说明                                                              |
| --------------------------------------------------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------- |
| `HOST` / `PORT`                                                                                           | `0.0.0.0` / `8993`         | 监听地址与端口（`python -m app.main` 无命令行参数，均由配置提供） |
| `INTERNAL_SERVICE_KEY`                                                                                    | `local-no-key`             | 服务间共享密钥，须与编排侧一致；pre/prod 在 `.env` 覆盖           |
| `MODEL_CACHE_DIR`                                                                                         | `<仓库根>/models`          | 模型持久化根目录（ASR 与 TTS 共用）                               |
| `TTS_MODEL_DIR`                                                                                           | `${MODEL_CACHE_DIR}/piper` | TTS 权重目录（`voice_params.onnx` 的相对路径基准）                |
| `ASR_DEVICE`                                                                                              | `auto`                     | `auto` 按 torch CUDA 可用性选择，也可显式 `cpu` / `cuda:0`        |
| `ASR_INFERENCE_THREADS` / `TTS_INFERENCE_THREADS`                                                         | `2` / `2`                  | 各自推理线程池大小（ASR 与 TTS 隔离）                             |
| `ASR_STREAM_IDLE_TIMEOUT`                                                                                 | `30`                       | 流式会话无输入自动收尾秒数（兜底，非前端契约）                    |
| `TTS_MAX_TEXT_CHARS`                                                                                      | `10000`                    | 单次合成文本上限（服务侧自限，编排侧另有校验）                    |
| `LOG_DIR` / `LOG_LEVEL` / `LOG_ENABLE_FILE` / `LOG_CONSOLE_JSON` / `LOG_RETENTION_DAYS` / `LOG_MAX_BYTES` | 见 `app/config.py`         | 日志目录、级别、开关、保留与切割（100MB 分片）                    |

配置由 `pydantic-settings` 读取仓库根 `.env`，进程环境变量优先。
ASR 的 ModelScope 缓存目录经 `MODELSCOPE_CACHE` 固定为 `${MODEL_CACHE_DIR}/modelscope`（运维显式设置该变量时优先生效）。

## 模型与权重

- **ASR**：`model_id` 即 ModelScope 模型 ID，首次请求时经 funasr `AutoModel` 下载并加载（懒加载单例）；
  权重落 `${MODEL_CACHE_DIR}/modelscope/hub/{命名空间}/{模型}`，重启不重复下载。部署时该目录需持久化卷。
- **TTS**：权重来源完全由请求 `voice_params` 下发（`onnx` 本地路径/相对路径/下载 URL、`size` 期望字节数、
  `urls` 镜像模板 `{filename}` 占位），引擎不查库、不做业务缓存：
  - `onnx` 落 `${TTS_MODEL_DIR}/<文件名>`，缺 `.json` 时按同一模板追加下载；
  - 跨进程文件锁（fcntl，Windows 退化为无锁）+ 断点续传（续传临时文件按进程隔离）+ 字节数校验；
  - 已就绪的文件不重复下载；并发下载时后落地者丢弃结果，避免覆盖正被推理占用的权重。
- 模型均在**首次请求**时加载，不阻塞 `/ready`；`/v1/models` 与 `/v1/status` 可查已加载/已缓存与设备。

## 测试

```bash
uv run pytest          # 61 项：离线/流式识别、EOS 收尾、热词按会话、TTS 编码与重采样、
                       # 权重下载与锁、探针语义、鉴权、日志字段契约
```

引擎在单测中全部桩化（ASR 转写与 TTS 模型加载），无需下载权重即可运行；
`test_tts.py` 中真实 lameenc / torchaudio 用例在依赖存在时才执行。

## 排障

| 现象                 | 处理                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `/ready` 503         | 看返回的 `message`：`funasr 不可用` / `piper-tts 不可用` → 在服务目录执行 `uv sync`            |
| 401 / WS 握手被拒    | 编排侧 `INTERNAL_SERVICE_KEY` 与本服务不一致（默认 `local-no-key`）                            |
| WS 4409              | 同一 `session_id` 已在进行中：编排侧重复建连或上次会话未收尾                                   |
| WS 4503              | 模型加载失败（`model_id` 不存在 / 权重损坏 / 磁盘不可写），日志已记录原因                      |
| 首次识别长时间无响应 | 首次请求触发 ModelScope 下载（数百 MB），确认 `MODEL_CACHE_DIR` 有持久化卷与写权限             |
| TTS 首字慢           | 首次合成需加载 onnx 权重（约 60MB）；确认 `${TTS_MODEL_DIR}` 有缓存且未被反复清理              |
| 想用 GPU 合成        | 安装 `onnxruntime-gpu` 替换 CPU 版；ASR 侧设置 `ASR_DEVICE=cuda:0` 并确保 torch 为 CUDA 版     |
| 权重下载失败         | 检查 `voice_params.urls` 镜像可达性；日志含每个镜像的失败原因，最终以 `MODEL_UNAVAILABLE` 返回 |
