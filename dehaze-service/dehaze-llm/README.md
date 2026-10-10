# dehaze-llm

本地轻量 LLM 推理服务（OpenAI 兼容），承载**对话 / 向量 / 重排**三类推理，端口 **8992**。

- 内网专用：调用方是 `dehaze-python`（编排侧）与 Prometheus，不直面 java/go 与前端
- 零业务状态：不连接 MySQL / Redis / ES / MinIO，不感知用户身份、配额与积分
- 独立生命周期：由 `scripts/run.py` 拉起（`run.py run llm`），与业务进程解耦——
  业务进程 reload/重启不影响推理可用性，多 Worker 下只有这一个模型服务实例

## 启动

```bash
cd dehaze-service/dehaze-llm
uv sync                                   # 首次：创建 .venv 并安装依赖
uv run python -m app.main                 # 前台启动（生产由 scripts/run.py 管理）

# 仓库根目录统一编排（推荐）
python scripts/run.py run llm
python scripts/run.py ps
python scripts/run.py logs llm 100
```

首次启动会后台下载模型（对话 378MB + 向量 610MB）到 `MODEL_CACHE_DIR`（默认仓库根 `models/`），
下载不阻塞进程启动，但**预热完成前 `/ready` 返回 503**。下载完成后落盘持久化，重启不再下载。

## 环境变量

配置走仓库根 `.env`（环境变量优先，pydantic-settings），变量名与编排侧同名。

| 变量                                                        | 默认值                | 说明                                                       |
| ----------------------------------------------------------- | --------------------- | ---------------------------------------------------------- |
| `HOST` / `PORT`                                             | `127.0.0.1` / `8992`  | 监听地址与端口                                             |
| `INTERNAL_SERVICE_KEY`                                      | `local-no-key`        | 入站鉴权密钥，须与编排侧同值                               |
| `MODEL_CACHE_DIR`                                           | 仓库根 `models/`      | 模型文件目录                                               |
| `MODEL_PATH` / `EMBEDDING_MODEL_PATH` / `RERANK_MODEL_PATH` | 空（取默认文件名）    | 各模型文件路径覆盖                                         |
| `CTX_SIZE`                                                  | `16384`               | 对话与重排上下文长度，须与编排侧 `LOCAL_LLM_CTX_SIZE` 一致 |
| `THREADS`                                                   | `0`（自动）           | 推理线程数                                                 |
| `NGPU_LAYERS`                                               | 空（自动探测）        | GPU 卸载层数；CUDA 构建自动全量卸载，纯 CPU 构建为 0       |
| `SERVICE_NAME`                                              | `llm`                 | 日志 `fields.service`                                      |
| `LOG_LEVEL` / `LOG_DIR`                                     | `INFO` / 服务 `logs/` | 日志级别、目录                                             |
| `LOG_MAX_BYTES` / `LOG_RETENTION_DAYS`                      | `100MB` / `30`        | 单文件切分阈值、按日期目录的保留期（`0` 表示不清理）       |

## 端点

全部端点校验 `Authorization: Bearer ${INTERNAL_SERVICE_KEY}`（缺失/错误 401）。

| 端点                   | 方法 | 说明                                                                                                      |
| ---------------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/v1/chat/completions` | POST | 对话，`stream=true` 走 SSE：逐 token 块 → `finish_reason` 尾块 → usage 块（`choices=[]`）→ `data: [DONE]` |
| `/v1/embeddings`       | POST | `input` 支持单条与批量（Qwen3-Embedding-0.6B，1024 维）                                                   |
| `/v1/rerank`           | POST | `query` + `documents` + 可选 `top_n`，按 `relevance_score` 降序                                           |
| `/v1/models`           | GET  | 能力发现：chat / embedding / rerank 的可用性与限制                                                        |
| `/v1/models/{id}`      | GET  | 单项详情                                                                                                  |
| `/health`              | GET  | liveness，恒 200，纯只读（不触发加载）                                                                    |
| `/ready`               | GET  | 对话与向量模型预热完成 200，否则 503（只读判定）                                                          |
| `/metrics`             | GET  | Prometheus 指标（`dehaze_http_*`、`dehaze_inference_*`）                                                  |

错误响应统一 `{"code": "<常量>", "message": "<可读信息>"}`（FastAPI 自身抛出的 404/405 等同样如此）：
400 `INVALID_REQUEST` / 404 `MODEL_NOT_FOUND`（未知路由为 `NOT_FOUND`）/ 500 `INFERENCE_FAILED` /
503 `MODEL_NOT_READY`。

行为契约：对话/向量/重排共用一个推理锁（串行执行，排队可被客户端断连取消）；
流式对话客户端断开即停止生成；`<think>` 思考块由消费端（编排侧 openai_compat）跨 chunk 识别，
本服务只剥离模板残留。

请求可观测：全部请求（含 401）计入 `dehaze_http_requests_total` 并落一条 ACCESS 日志
（`fields.service=llm`，logger `app.request`，含 `method`/`path`/`status`/`duration_ms`）；
`/health`、`/ready`、`/metrics` 只计数不落 ACCESS 日志（避免巡检流量淹没业务请求）；
编排侧透传的 `X-Trace-Id` 写入日志上下文并回写响应头。

**为何本服务的 trace/ACCESS 中间件用纯 ASGI（`app/middleware.py`）而非另两个服务的
`BaseHTTPMiddleware`**：`BaseHTTPMiddleware` 会在响应外层再包一层任务组，对 SSE 流式响应的
断连取消语义有干扰，而流式对话必须保持"客户端断连即停止生成"；`dehaze-voice` 的核心是
WebSocket（不经 HTTP 中间件链）、`dehaze-modal` 是一次性字节响应，二者无此约束。

## 模型

| 模型                   | 文件                             | 用途            |        预热        |
| ---------------------- | -------------------------------- | --------------- | :----------------: |
| `qwen3-0.6b`           | `Qwen3-0.6B-Q4_K_M.gguf`         | 对话（含流式）  |         是         |
| `qwen3-embedding-0.6b` | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 向量（1024 维） |         是         |
| `qwen3-reranker-0.6b`  | `Qwen3-Reranker-0.6B-q8_0.gguf`  | 重排（可选）    | 否，首次请求懒下载 |

下载实现：hf-mirror → huggingface 镜像回退、Range 断点续传、`fcntl` 跨进程文件锁、
结束字节数校验后原子替换生效。手动预下载（如离线环境提前准备重排模型）：

```bash
cd dehaze-service/dehaze-llm
uv run python -c "from app.model_store import ensure_rerank_model; ensure_rerank_model()"
```

## 测试

```bash
cd dehaze-service/dehaze-llm
uv run pytest          # 57 项：鉴权（含空密钥 fail-closed）、探针语义、对话/向量/重排契约、
                       # 能力发现、流式 usage 与断连即停、错误信封、指标与日志字段
```

`llama.cpp` 在单测中以桩替换（`tests/conftest.py`），无需真实 GGUF 模型与 GPU；
模型下载用例以 `respx` 拦截 HTTP。

## 排障

| 现象                  | 排查                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------- |
| `/ready` 长期 503     | 看 `logs/{今日}/console.log` 的"预热失败"日志（网络不可达 hf-mirror / 磁盘空间）；模型预热失败每 30s 自动重试 |
| 401 未授权            | 服务与编排侧 `INTERNAL_SERVICE_KEY` 不一致；`Authorization` 必须是 `Bearer <key>`                             |
| 503 `MODEL_NOT_READY` | 该模型文件缺失（未下载或大小不符），确认 `MODEL_CACHE_DIR` 与 `models/` 内容                                  |
| 重排首次请求 503      | 属预期：已触发后台下载，约 610MB 完成后重试即可                                                               |
| 推理慢/首字延迟高     | 确认是否跑在纯 CPU（日志有"GPU 卸载能力探测失败"警告），或 `THREADS` 配置过小                                 |
| 日志位置              | `logs/{yyyy-MM-dd}/{console,info,error}.log`（NDJSON，`fields.service=llm`）                                  |
