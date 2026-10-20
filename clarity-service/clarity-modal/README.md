# clarity-modal

图像推理执行器服务（端口 **8994**）：承载图像编辑推理执行、图像质量指标计算、模型权重缓存与 GPU 指标。
服务边界与契约见 [AI 能力链路协作设计](../../clarity-doc/docs/02-系统架构/09-AI能力链路协作设计.md)。

- 只做「模型标识 + 参数进，结果图/指标出」，**零 MySQL / Redis / ES / MinIO 依赖**，无业务状态，可无状态重启
- 状态机、任务编排、配额、结果存储与命名、缓存与拦截器全部留在编排侧（`clarity-python`）
- 图像**字节进、字节出**：不按 URL 拉图，不持有存储凭证
- 模型标识取统一模型表（`sys_ai_model`）的规范 slug，import_path / 权重相对路径由本服务目录（`app/catalog.py`）解析，编排侧不下发运行工件

## 启动

```bash
cd clarity-service/clarity-modal
uv sync                      # 首次安装依赖（torch CUDA 版，体积较大）
uv run python -m app.main    # 默认 0.0.0.0:8994
```

生产/联调由仓库根 `scripts/run.py` 统一拉起（`uv sync` + `.venv/bin/python -m app.main`）。
日志写入服务目录 `logs/{yyyy-MM-dd}/info.log|error.log`（NDJSON，`fields.service=modal`，
按大小分片 + 默认保留 30 天），由 filebeat 采集，无需手工清理。

容器化：`docker build -t clarity-modal .`（CUDA + cuDNN 基座，含 DCN 算子 JIT 编译所需的 `nvcc`/`ninja`；
`MODEL_CACHE_DIR=/models` 与 `TORCH_EXTENSIONS_DIR` 建议挂持久卷，重启不重复下载/编译）。

## 端点

所有端点（含 `/health`、`/ready`、`/metrics`）都要求 `Authorization: Bearer ${INTERNAL_SERVICE_KEY}`，
缺失或错误返回 401；`INTERNAL_SERVICE_KEY` 未配置（空串）时**全部 401**（fail closed，不因空密钥退化为无鉴权）。
鉴权由纯 ASGI 中间件在最外层完成，`401` 同样带 `X-Trace-Id` 与 ACCESS 日志。
`/docs`、`/redoc`、`/openapi.json` 已关闭（仅服务间调用）。

| 端点               | 方法              | 请求                                                                                   | 响应                                                                                                                |
| ------------------ | ----------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `/v1/images/edits` | POST（multipart） | `image`（原图字节）、`model`（统一模型规范 slug，如 `aecrnet`）、`params`（可选 JSON） | `{"created":<ts>,"data":[{"b64_json":"..."}]}`（结果图，b64 编码）                                                  |
| `/v1/metrics`      | POST（multipart） | `image`、`reference`（可选）、`model`（指标集标识）、`metrics`（JSON 数组）            | `{"metrics":{name:value},"failed":[name],"meta":{image_size,width,height}}`                                         |
| `/v1/models`       | GET               | —                                                                                      | `{"data":[...]}`：图像编辑模型（`type=image_edit`）+ 指标集（`type=metrics`）+ 服务自带算法实现（`type=inference`） |
| `/v1/models/{id}`  | GET               | `id` = 服务自带算法 import_path 或权重相对路径                                         | 单项详情；权重路径返回 `available` 与 `size_bytes`（权重可用性探测）                                                |
| `/health`          | GET               | —                                                                                      | `{"status":"ok"}`，恒 200，不触发任何加载                                                                           |
| `/ready`           | GET               | —                                                                                      | 设备与权重来源可用 → 200；否则 503（只读判定，不触发下载/加载）                                                     |
| `/metrics`         | GET               | —                                                                                      | Prometheus 指标（`dehaze_http_*`、`dehaze_inference_*`、`dehaze_gpu_*`）                                            |

图像编辑模型的 `type` 取 `image_edit`，`id` 为统一模型表规范 slug，`artifacts` 内嵌
`import_path` / `weights_path`，是编排侧模型驱动推理的清单事实源。权重文件不在服务目录
（懒下载，见 `algorithm/model_loader.py`），可用性由 `GET /v1/models/{权重相对路径}` 的
`available` / `size_bytes` 探测。

请求示例：

```bash
curl -X POST http://127.0.0.1:8994/v1/images/edits \
  -H "Authorization: Bearer local-no-key" \
  -F "image=@hazy.png" \
  -F "model=aecrnet"

# 响应（OpenAI Images 风格，b64 载荷）：
# {"created":1759100000,"data":[{"b64_json":"iVBORw0KGgo..."}]}

curl -X POST http://127.0.0.1:8994/v1/metrics \
  -H "Authorization: Bearer local-no-key" \
  -F "image=@result.png" -F "reference=@gt.png" \
  -F "model=指标集A" -F 'metrics=["psnr","ssim","niqe"]'
```

## 错误模型（§3.1）

统一响应体 `{"code": "<大写下划线常量>", "message": "<可读信息>"}`：

| 状态 | code                                  | 语义                                                                                                 |
| :--: | ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 400  | `PARAM_ERROR`                         | 表单字段缺失/JSON 格式错误/未知指标名/预测图与参考图尺寸不一致                                       |
| 401  | `UNAUTHORIZED`                        | 内部服务密钥缺失或错误                                                                               |
| 404  | `NOT_FOUND` / `MODEL_NOT_FOUND`       | 服务自带算法标识不存在（模块未随服务发布）/ 统一模型 slug 不在目录中                                 |
| 404  | `MODEL_DISABLED`                      | 模型已停用（目录级 disabled，拒绝承接推理）                                                          |
| 422  | `INVALID_IMAGE`                       | 图像内容非法（无法解码）                                                                             |
| 500  | `INFERENCE_FAILED` / `INTERNAL_ERROR` | 模型执行失败（message 为截断后的失败摘要）/ 未包装异常（message 固定「内部错误」，异常原文只进日志） |
| 503  | `MODEL_UNAVAILABLE`                   | 权重不可用（本地无缓存且下载源不可达）                                                               |
| 503  | `SERVICE_UNAVAILABLE`                 | `/ready` 未就绪（缓存目录不可写、推理依赖不可用）                                                    |
| 504  | `INFERENCE_TIMEOUT`                   | 推理超过 `INFERENCE_TIMEOUT_SECONDS`（默认 30s，线程不可中断，放弃等待结果）                         |

## 环境变量

配置读仓库根 `.env`，环境变量优先（`app/config.py`）。

| 变量                                      | 默认                       | 说明                                                                |
| ----------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `APP_NAME` / `SERVICE_NAME`               | `Dehaze Modal` / `modal`   | 服务名；`SERVICE_NAME` 用于日志 `fields.service`                    |
| `HOST` / `PORT`                           | `0.0.0.0` / `8994`         | 监听地址与端口                                                      |
| `INTERNAL_SERVICE_KEY`                    | `local-no-key`             | 内部服务密钥（pre/prod 在 `.env` 覆盖；置空则全部 401 fail closed） |
| `INFERENCE_THREAD_POOL_SIZE`              | `2`                        | 推理线程池容量，按单卡显存配置；超限请求排队不丢弃                  |
| `INFERENCE_TIMEOUT_SECONDS`               | `30`                       | 单次推理超时，超时返回 504                                          |
| `MODEL_CACHE_DIR`                         | 仓库根 `models/`           | 权重缓存目录（需持久化，避免重启重下）                              |
| `NGINX_STATIC_HOST` / `NGINX_STATIC_PORT` | `127.0.0.1` / `19000`      | 权重下载源（`MODEL_BASE_URL` = `http://{host}:{port}/models`）      |
| `GPU_METRICS_INTERVAL_SECONDS`            | `5`                        | GPU 指标采集间隔                                                    |
| `LOG_LEVEL` / `LOG_DIR`                   | `INFO` / `<服务目录>/logs` | 日志级别、目录                                                      |
| `LOG_MAX_BYTES` / `LOG_RETENTION_DAYS`    | `100MB` / `30`             | 单文件切分阈值、按日期目录的保留期（`0` 表示不清理）                |

## GPU 与权重

- **设备选择**：`torch.cuda.is_available()` 为真用 `cuda:0`，否则 CPU 回退（`algorithm/config.py` 的 `Config.DEVICE`，`/ready` 的 `device` 字段可见）
- **显存独占**：CUDA 上下文只存在于本服务；CPU/GPU 分工见设计 §6.10
- **权重来源**：优先 `MODEL_CACHE_DIR/{relative_path}` 命中即用（不进网络，即降级口径）；未命中从 `MODEL_BASE_URL` 下载，下载走「临时文件 + 原子替换」，失败返回 503 而非使用半截文件
- **不做模型实例缓存**：每次推理重新 `torch.load`（§6.1 现状口径）；权重目录挂持久卷可避免重启后重新下载
- **首次加载**：`RIDCP`/`WPXNet`/`CFENViTDehazing` 的 DCN 算子在首次使用时会 JIT 编译 CUDA 扩展（依赖 `nvcc`、`ninja`，缓存目录 `TORCH_EXTENSIONS_DIR`）

## 图像模型与算法实现

`algorithm/` 为迁移进来的 31 种算法实现，每个算法目录的 `run.py` 导出 `dehaze(image, model_path)`。
算法内部一律使用 `algorithm.xxx` 绝对导入，因此**必须以服务目录为工作目录启动**（`python -m app.main`）。

`algorithm/__init__.py` 不做任何导入：权重可用性探测（`/v1/models/{权重路径}`）与能力发现
不需要拉起 torch/thop，也不应产生额外 CUDA 上下文。

图像模型清单来自统一模型表（`sys_ai_model` 中 `model_type='image_edit'` 的 `model_id`），
slug → (import_path, weights_path) 映射由 `app/catalog.py` / `app/artifacts.json` 静态承载，服务侧不查库。

模型验收（L1 权重可解析 / L2 权重可反序列化 / L3 结构匹配 / L4 冒烟推理）：

```bash
uv run scripts/validate_algorithms.py --levels L1,L2,L3,L4 --output report.json
```

## 测试

```bash
uv run pytest            # 48 例：鉴权（含空密钥 fail-closed）、探针与 HTTP 指标、图像编辑与指标契约、权重解析、能力发现
```

指标计算用例以假指标模型注入，不触发 pyiqa 权重下载；权重下载用例以 respx 拦截 HTTP。

## 排障

| 现象                                  | 排查                                                                                                                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/ready` 503 且提示缓存目录不可写     | 检查 `MODEL_CACHE_DIR` 挂载与权限（容器/宿主用户一致）                                                                                                             |
| 推理 503 `MODEL_UNAVAILABLE`          | slug 的权重相对路径是否正确、nginx `/models` 是否可达、宿主是否能访问 `MODEL_BASE_URL`                                                                             |
| 推理 504                              | 模型过大/图片分辨率过高；调大 `INFERENCE_TIMEOUT_SECONDS` 前先确认编排侧 30s 超时口径                                                                              |
| `/v1/models` 中模型 `available=false` | `image_edit` 项为目录级 disabled；`inference` 项为算法目录缺少 `run.py` 或未导出 `dehaze()`                                                                        |
| `/metrics` 无 `dehaze_gpu_*`          | 无 NVIDIA 驱动或 `pynvml` 初始化失败，日志中可见「GPU 指标采集器初始化失败」                                                                                       |
| 401                                   | 编排侧是否携带 `Authorization: Bearer ${INTERNAL_SERVICE_KEY}`；Prometheus job 需配置 `authorization.credentials`；密钥为空串时本服务会拒绝全部请求（fail closed） |
| 模型首次推理卡住                      | DCN 算子 JIT 编译耗时（`nvcc`/`ninja` 是否可用），编译缓存是否可写                                                                                                 |
