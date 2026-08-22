from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置（多环境，通过 APP_ENV 切换）"""

    model_config = SettingsConfigDict(
        # 项目根目录（config.py 位于 app/，向上三级），.env 文件位于此处
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 应用与环境 =====
    APP_NAME: str = "Dehaze API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # uvicorn 服务配置（仅 `python -m app.main` 启动方式生效）
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = Field(default=8991, gt=0, le=65535)
    SERVER_WORKERS: int = Field(default=4, gt=0)

    # ===== 基础设施统一凭证 =====
    # 统一主机地址，DB/Redis/MongoDB/MinIO/RabbitMQ/XXL-Job 地址均从此派生
    DEHAZE_HOST: str = Field(default="127.0.0.1")
    # 统一密码，复用为 DB/Redis/MongoDB/MinIO/RabbitMQ/XXL-Job 凭证
    DEHAZE_PASSWORD: str = Field(default="")

    # ===== 数据库 =====
    DB_PORT: int = Field(default=3306, gt=0, le=65535)
    DB_NAME: str = "dehaze"
    DB_USER: str = "root"
    DATABASE_POOL_SIZE: int = Field(default=10, gt=0)
    DATABASE_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DATABASE_POOL_RECYCLE: int = Field(default=3600, gt=0)
    # SQL 审计日志级别：INFO 记录全部 SQL（开发/测试），WARNING 仅记慢查询与错误（生产）
    SQL_LOG_LEVEL: Literal["INFO", "WARNING", "ERROR"] = "INFO"
    # 慢查询阈值（毫秒），超过则额外输出 WARNING 级 SLOW_SQL
    SQL_SLOW_THRESHOLD_MS: int = Field(default=500, gt=0)

    @property
    def DB_HOST(self) -> str:
        return self.DEHAZE_HOST

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DEHAZE_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"

    # ===== Redis =====
    REDIS_PORT: int = Field(default=6379, gt=0, le=65535)
    REDIS_DB: int = Field(default=0, ge=0)
    REDIS_MAX_CONNECTIONS: int = Field(default=100, gt=0)
    REDIS_SOCKET_TIMEOUT: float = Field(default=5.0, gt=0)
    REDIS_SOCKET_CONNECT_TIMEOUT: float = Field(default=5.0, gt=0)
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_HEALTH_CHECK_INTERVAL: int = Field(default=30, gt=0)

    @property
    def REDIS_HOST(self) -> str:
        return self.DEHAZE_HOST

    @property
    def REDIS_URL(self) -> str:
        return (
            f"redis://:{self.DEHAZE_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        )

    # ===== MongoDB（审计日志） =====
    MONGO_DB_NAME: str = "dehaze"

    @property
    def MONGO_URI(self) -> str:
        return f"mongodb://root:{self.DEHAZE_PASSWORD}@{self.DEHAZE_HOST}:27017/{self.MONGO_DB_NAME}?authSource=admin"

    # ===== 文件与模型存储 =====
    MINIO_ACCESS_KEY: str = "admin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "dehaze"
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024, gt=0)
    # 上传/删除文件使用的默认存储后端（minio/local/nginx-static）
    FILE_STORAGE_TYPE: Literal["minio", "local", "nginx-static"] = "minio"
    LOCAL_STORAGE_PATH: str = "/data/files"
    FILE_TEMP_CLEANUP_HOURS: int = Field(default=24, gt=0)
    # 统一模型文件目录（算法权重缓存 / LLM / embedding / TTS 模型均存放于此）
    MODEL_CACHE_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "models")
    MODEL_FALLBACK_TO_LOCAL: bool = True
    TEMP_DIR: str = ""

    @property
    def MINIO_ENDPOINT(self) -> str:
        """MinIO 服务端点（host:port，不含 scheme，供 MinIO SDK 使用）"""
        return f"{self.DEHAZE_HOST}:9110"

    @property
    def MINIO_SECRET_KEY(self) -> str:
        return self.DEHAZE_PASSWORD

    @property
    def MODEL_BASE_URL(self) -> str:
        return f"http://{self.DEHAZE_HOST}:9000/models"

    # ===== 存储后端 baseUrl（运行时拼接 URL 用，必须是完整 URL，禁止相对路径）=====
    @property
    def FILE_STORAGE_BASE_URLS(self) -> dict[str, str]:
        """各存储后端的 baseUrl 映射，用于运行时拼接完整 URL。

        - minio：MinIO 直连地址（bucket 已设为 public read），三端可直接 HTTP 访问
        - local：本地存储文件只能通过 Java 下载接口访问
        - nginx-static：nginx 静态服务根地址（不带 /datasets、/models 等资源子路径），
          object_name 自带资源前缀（datasets/...、models/...）
        """
        return {
            "minio": f"http://{self.MINIO_ENDPOINT}/{self.MINIO_BUCKET_NAME}",
            "local": f"http://{self.DEHAZE_HOST}:8989/api/v1/files/download",
            "nginx-static": f"http://{self.DEHAZE_HOST}:9000",
        }

    @property
    def TEMP_DIR_RESOLVED(self) -> str:
        """未设置 TEMP_DIR 时使用系统临时目录"""
        import tempfile

        return self.TEMP_DIR if self.TEMP_DIR else tempfile.gettempdir()

    # ===== RabbitMQ =====
    RABBITMQ_ENABLED: bool = False
    RABBITMQ_PORT: int = Field(default=5672, gt=0, le=65535)
    RABBITMQ_USER: str = "guest"
    RABBITMQ_EXCHANGE: str = "dehaze.tasks"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_RECONNECT_MAX_RETRIES: int = Field(default=0, ge=0)  # 0 表示无限重试
    RABBITMQ_RECONNECT_INITIAL_INTERVAL: float = Field(default=1.0, gt=0)
    RABBITMQ_RECONNECT_MAX_INTERVAL: float = Field(default=30.0, gt=0)
    RABBITMQ_PREFETCH_COUNT: int = Field(default=2, gt=0)
    RABBITMQ_RETRY_DELAYS: list[int] = [5000, 30000, 300000]  # 分级重试延迟（ms）: 5s/30s/5min

    @property
    def RABBITMQ_HOST(self) -> str:
        return self.DEHAZE_HOST

    @property
    def RABBITMQ_PASSWORD(self) -> str:
        """未设置 DEHAZE_PASSWORD 时返回开发默认值 guest"""
        return self.DEHAZE_PASSWORD or "guest"

    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/%2F"

    # ===== XXL-Job 定时任务 =====
    XXLJOB_ENABLED: bool = False
    XXLJOB_EXECUTOR_APP_NAME: str = "xxl-job-executor-dehaze-python"
    XXLJOB_EXECUTOR_HOST: str = "0.0.0.0"
    XXLJOB_EXECUTOR_PORT: int = Field(default=9998, gt=0, le=65535)
    XXLJOB_TASK_LOG_DIR: str = "logs/xxljob-tasks"
    XXLJOB_PID_FILE: str = "logs/pyxxl.pid"
    # 留空则复用 DEHAZE_PASSWORD（由 model_validator 处理）
    XXLJOB_ACCESS_TOKEN: str = ""

    @property
    def XXLJOB_ADMIN_URL(self) -> str:
        return f"http://{self.DEHAZE_HOST}:14980/xxl-job-admin/api/"

    # ===== 日志 =====
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s - %(levelname)s [%(trace_id)s] --- [%(thread)d] %(name)s : %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_DIR: str = "logs"
    LOG_RETENTION_DAYS: int = Field(default=30, gt=0)
    # 单个日志文件大小上限（字节），超限归档为 {级别}.{n}.log 并开新活动文件，0 表示不按大小切割
    LOG_MAX_BYTES: int = Field(default=100 * 1024 * 1024, ge=0)
    LOG_ENABLE_CONSOLE: bool = True
    LOG_ENABLE_FILE: bool = True
    LOG_FORMAT_JSON: bool = False
    # 启动时归档当天已存在的活动日志文件（dev 用，prod 关闭以保留连续日志）
    LOG_ARCHIVE_ON_STARTUP: bool = False

    # ===== Prometheus 监控 =====
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_GPU_COLLECT_INTERVAL: int = Field(default=5, gt=0)
    # 设置后 /metrics 端点通过 MultiProcessCollector 聚合所有 Worker 指标
    PROMETHEUS_MULTIPROC_DIR: str = ""

    # ===== 安全与认证 =====
    # 验证码
    CAPTCHA_KEY_PREFIX: str = "captcha_code:"
    CAPTCHA_LENGTH: int = Field(default=4, ge=4)
    CAPTCHA_WIDTH: int = Field(default=120, gt=0)
    CAPTCHA_HEIGHT: int = Field(default=40, gt=0)
    CAPTCHA_FONT_SIZE: int = Field(default=24, gt=0)
    CAPTCHA_NOISE_LINES: int = Field(default=5, ge=0)
    CAPTCHA_EXPIRES: int = Field(default=300, gt=0)

    # Session Cookie
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_PATH: str = "/api"
    USE_MULTI_POINT: bool = False

    # IP 黑名单
    IP_BLACKLIST_ENABLED: bool = True
    IP_BLACKLIST_THRESHOLD: int = Field(default=100, gt=0)
    IP_BLACKLIST_DURATION: int = Field(default=3600, gt=0)
    IP_BLACKLIST_TRACKING_WINDOW: int = Field(default=60, gt=0)

    # 登录失败锁定
    LOGIN_FAIL_MAX_ATTEMPTS: int = Field(default=5, gt=0)
    LOGIN_FAIL_LOCK_MINUTES: int = Field(default=30, gt=0)

    # 限流
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, gt=0)
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=60, gt=0)

    # 优惠券领取限流（每分钟）
    COUPON_RECEIVE_RATE_LIMIT: int = Field(default=5, gt=0)
    COUPON_RECEIVE_RATE_WINDOW: int = Field(default=60, gt=0)

    # 防重复提交
    ANTI_REPEAT_ENABLED: bool = True
    ANTI_REPEAT_TTL_SECONDS: int = Field(default=5, gt=0)

    # 密码策略
    PASSWORD_MIN_LENGTH: int = Field(default=8, gt=0)
    PASSWORD_REQUIRE_COMPLEXITY: bool = True

    # CORS 跨域
    # 开发环境白名单（端口规范：5173 React / 5174 Vue / 5175 Taro / 5176 uniapp /
    # 5177 Flutter Web / 5183 React Electron / 5184 Vue Electron / 8081 RN Metro）
    CORS_ORIGINS: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://localhost:5183",
        "http://localhost:5184",
        "http://localhost:8081",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
        "http://127.0.0.1:5183",
        "http://127.0.0.1:5184",
        "http://127.0.0.1:8081",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """支持环境变量传入逗号分隔字符串或 JSON 数组"""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ===== WebSocket 跨 Worker =====
    WS_REDIS_CHANNEL: str = "dehaze:ws:broadcast"
    WS_ONLINE_KEY: str = "dehaze:ws:online_users"
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, gt=0)
    WS_ONLINE_TTL: int = Field(default=90, gt=0)  # 应 >= 3 倍心跳间隔

    # ===== TaskTracker 跨 Worker =====
    TASK_REDIS_KEY_PREFIX: str = "task:running"
    TASK_HEARTBEAT_INTERVAL: int = Field(default=30, gt=0)
    TASK_REDIS_TTL: int = Field(default=3600, gt=0)
    GRACEFUL_SHUTDOWN_TIMEOUT: int = Field(default=30, gt=0)

    # ===== 算法推理 =====
    # PyTorch 推理线程池并发数，按 GPU 显存/卡数调整（单卡 24GB 建议 1-2）
    INFERENCE_THREAD_POOL_SIZE: int = Field(default=2, gt=0)

    # ===== 缓存 =====
    CACHE_L1_ENABLED: bool = True
    CACHE_L1_MAXSIZE: int = Field(default=1000, gt=0)
    CACHE_L1_TTL: int = Field(default=300, gt=0)
    CACHE_SINGLEFLIGHT_ENABLED: bool = True
    CACHE_NULL_ENABLED: bool = True
    CACHE_NULL_TTL: int = Field(default=60, gt=0)
    CACHE_INVALIDATION_CHANNEL: str = "cache:invalidation"

    # ===== 业务配置 =====
    # 新用户默认密码，留空则复用 DEHAZE_PASSWORD（由 model_validator 处理）
    DEFAULT_PASSWORD: str = ""

    # 支付渠道 — 微信
    PAYMENT_WECHAT_ENABLED: bool = False
    PAYMENT_WECHAT_APP_ID: str = ""
    PAYMENT_WECHAT_MCH_ID: str = ""
    PAYMENT_WECHAT_API_V3_KEY: str = ""
    PAYMENT_WECHAT_CERT_SERIAL_NO: str = ""
    PAYMENT_WECHAT_PRIVATE_KEY_PATH: str = ""
    PAYMENT_WECHAT_NOTIFY_URL: str = ""
    PAYMENT_WECHAT_REFUND_NOTIFY_URL: str = ""
    PAYMENT_WECHAT_BASE_URL: str = "https://api.mch.weixin.qq.com"

    # 支付渠道 — 支付宝
    PAYMENT_ALIPAY_ENABLED: bool = False
    PAYMENT_ALIPAY_APP_ID: str = ""
    PAYMENT_ALIPAY_PRIVATE_KEY: str = ""
    PAYMENT_ALIPAY_PUBLIC_KEY: str = ""
    PAYMENT_ALIPAY_NOTIFY_URL: str = ""
    PAYMENT_ALIPAY_BASE_URL: str = "https://openapi.alipay.com/gateway.do"

    # 自动续费
    AUTO_RENEW_RETRY_MAX: int = Field(default=3, gt=0)
    AUTO_RENEW_RETRY_INTERVAL_HOURS: int = Field(default=2, gt=0)
    AUTO_RENEW_DISCOUNT: float = Field(default=0.95, gt=0, le=1)

    # ===== AI 对话 =====
    # 单条消息最大字符数
    AI_MESSAGE_MAX_LENGTH: int = 4000
    # 流式输出超时（秒）
    AI_MESSAGE_STREAM_TIMEOUT: int = 120
    # SSE 心跳间隔（秒）
    AI_MESSAGE_HEARTBEAT_INTERVAL: int = 15
    # 默认模型（内置本地轻量 LLM，Qwen3-0.6B，部署零外部依赖；模型清单由
    # infrastructure/llm/model_seeder 幂等播种，路由由 sys_ai_provider/sys_ai_model 注册表驱动，
    # 接入第三方/切换供应商只需在库内配置，无需改代码）
    AI_DEFAULT_MODEL: str = "qwen3-0.6b"
    # 内置本地模型服务（llama-cpp-python 子进程，OpenAI 兼容；LLM 对话 + Embedding 共用，
    # 端点由 local provider 的 api_base_url 派生：/v1/chat/completions 与 /v1/embeddings）
    LOCAL_LLM_HOST: str = "127.0.0.1"
    LOCAL_LLM_PORT: int = 8992
    LOCAL_LLM_MODEL_PATH: str = ""  # 空 = 默认 models/Qwen3-0.6B-Q4_K_M.gguf（对话）
    LOCAL_LLM_EMBEDDING_MODEL_PATH: str = ""  # 空 = 默认 models/Qwen3-Embedding-0.6B-Q8_0.gguf（向量）
    LOCAL_LLM_CTX_SIZE: int = 8192
    LOCAL_LLM_THREADS: int = 0  # 0 = 自动（物理核数）
    LOCAL_LLM_NGPU_LAYERS: int | None = None  # None = 自动（CUDA 构建全量卸载 -1，纯 CPU 构建 0）
    # 速度档位 P95 延迟阈值（毫秒）：P95 < fast 为快；< medium 为中；其余为慢
    AI_SPEED_TIER_FAST_P95_MS: int = 2000
    AI_SPEED_TIER_MEDIUM_P95_MS: int = 8000
    # 供应商 API Key 加密密钥（AES-256，环境变量注入）
    AI_PROVIDER_KEY_ENCRYPTION_KEY: str = ""
    # MCP 能力网关
    MCP_GATEWAY_URL: str = "http://127.0.0.1:8082/mcp"
    # 推理参数（引擎已切换 deepagents，旧 ReAct/Plan/Reflexion 分步参数废弃）
    AI_REASONING_TOOL_TIMEOUT: int = 60
    AI_REASONING_RETRY_MAX: int = 2
    # 能力扩展（F-M08-006）约束
    # 虚拟文件系统工作区容量上限（字节，默认 100MB）
    AI_VFS_MAX_BYTES: int = Field(default=100 * 1024 * 1024, gt=0)
    # 同一工具单次推理内连续失败剔除阈值
    AI_TOOL_CONSECUTIVE_FAIL_LIMIT: int = Field(default=3, gt=0)
    # 任务清单单次上限与单条描述长度引导阈值
    AI_TODOS_MAX_ITEMS: int = Field(default=32, gt=0)
    AI_TODOS_ITEM_MAX_CHARS: int = Field(default=50, gt=0)
    # MCP 命名空间摘要缓存时长（秒，对齐网关定时同步间隔）
    AI_MCP_NAMESPACE_CACHE_TTL: int = Field(default=300, gt=0)

    # ===== AI 定时调度（F-M08-009）=====
    # 平台级全局并发上限（防止任务风暴打爆推理服务）
    AI_SCHEDULE_MAX_CONCURRENT: int = Field(default=10, gt=0)
    # 单任务运行标记 TTL（秒）：防进程崩溃后 Redis 标记泄漏，亦作执行中批次僵尸回收阈值
    AI_SCHEDULE_RUN_MARK_TTL: int = Field(default=3600, gt=0)
    # 连续失败熔断阈值（>= 该值自动停用任务并通知）
    AI_SCHEDULE_CIRCUIT_THRESHOLD: int = Field(default=5, gt=0)
    # 临时错误指数退避重试次数（退避间隔 1s/2s/4s）
    AI_SCHEDULE_RETRY_MAX: int = Field(default=3, ge=0)

    # ===== AI 计费管理 =====
    # 新用户注册赠送试用积分
    AI_BILLING_TRIAL_CREDITS: int = 100
    # 异常检测：单次超高阈值（月限额百分比）
    AI_BILLING_ANOMALY_SINGLE_HIGH_THRESHOLD: float = 0.1
    # 异常检测：突发峰值阈值（日限额百分比）
    AI_BILLING_ANOMALY_BURST_THRESHOLD: float = 0.5
    # 异常检测：突发检测时间窗口（分钟）
    AI_BILLING_ANOMALY_BURST_WINDOW_MINUTES: int = 5
    # 异常检测：连续配额不足告警阈值（次/24h）
    AI_BILLING_ANOMALY_CONSECUTIVE_QUOTA_FAIL: int = 10
    # 月结账单 Redis 缓存 TTL（秒，90 天）
    AI_BILLING_BILL_CACHE_TTL: int = 90 * 24 * 3600

    # 批量处理
    AI_BATCH_MAX_PARALLEL: int = 5
    AI_BATCH_STRATEGY: str = "auto"
    AI_BATCH_MAX_IMAGES: int = 20
    # 批量处理异步阈值：超过该数量采用 async_wait 中断（提交后台任务 + 回调自动恢复），
    # 不超过保持同步直返，避免轻任务中断体验劣化
    AI_BATCH_ASYNC_THRESHOLD: int = 3
    # 长期记忆管理
    AI_MEMORY_MAX_COUNT: int = Field(default=200, gt=0)
    AI_MEMORY_FORGET_THRESHOLD: int = Field(default=10, ge=0)
    AI_MEMORY_HALF_LIFE_DAYS: int = Field(default=30, gt=0)

    # ===== AI 计费 =====
    # 预估输入 Token 保守系数（应用于当前消息 Token）
    AI_BILLING_ESTIMATE_INPUT_FACTOR: float = Field(default=1.2, gt=0)
    # 预估输出 Token 保守系数（max_output_tokens × 系数）
    AI_BILLING_ESTIMATE_OUTPUT_FACTOR: float = Field(default=0.3, gt=0, le=1)
    # 预估输入中的记忆注入 Token / 工具定义 Token
    AI_BILLING_ESTIMATE_MEMORY_TOKENS: int = 200
    AI_BILLING_ESTIMATE_TOOL_TOKENS: int = 500
    # 历史上下文均值滑动的窗口权重（新值占比）
    AI_BILLING_CTX_AVG_WEIGHT: float = Field(default=0.5, gt=0, le=1)

    # ===== Elasticsearch（记忆向量检索 / 会话全文检索，必选基础设施，docker-compose 统一部署） =====
    ES_URL: str = "http://localhost:9200"
    ES_API_KEY: str = ""  # 可选；与用户名密码二选一
    ES_USERNAME: str = "elastic"  # docker-compose 启用 xpack security，elastic 超级用户
    ES_PASSWORD: str = ""  # 留空则复用 DEHAZE_PASSWORD（由 model_validator 处理）

    # ===== AI 知识库 =====
    # 分块默认/边界
    KB_CHUNK_DEFAULT_SIZE: int = Field(default=800, gt=0, description="默认分块大小(token)")
    KB_CHUNK_DEFAULT_OVERLAP: int = Field(default=80, ge=0, description="默认分块重叠(token)")
    KB_CHUNK_MAX_TOKENS: int = Field(default=2000, gt=0, description="最大分块 token 数")
    KB_CHUNK_MIN_TOKENS: int = Field(default=50, gt=0, description="最小分块 token 数")
    # 检索参数
    KB_SEARCH_DEFAULT_TOP_K: int = Field(default=5, gt=0, description="默认检索 Top-K")
    KB_SEARCH_MAX_TOP_K: int = Field(default=50, gt=0, description="最大检索 Top-K")
    KB_SEARCH_DEFAULT_THRESHOLD: float = Field(
        default=0.5, gt=0, lt=1, description="默认相似度阈值"
    )
    KB_SEARCH_HYBRID_RANK_CONSTANT: int = Field(
        default=60, gt=0, description="RRF 混合检索 rank_constant"
    )
    KB_SEARCH_HYBRID_RANK_WINDOW: int = Field(
        default=20, gt=0, description="RRF 混合检索 rank_window_size"
    )
    KB_SEARCH_TIMEOUT_MS: int = Field(
        default=2000, gt=0, description="检索超时(ms)，超时降级无知识回复"
    )
    # 索引与异步处理
    KB_INDEX_BATCH_SIZE: int = Field(default=100, gt=0, description="ES 批量索引条数")
    KB_ASYNC_MAX_RETRY: int = Field(default=3, ge=0, description="异步处理最大重试次数")

    # ===== 语音交互 =====
    # FunASR 引擎进程内运行（CPU 推理：流式 SenseVoice-Small / 离线 SeACo-Paraformer-Large）
    VOICE_ASR_STREAM_MODEL: str = "sensevoice"
    VOICE_ASR_OFFLINE_MODEL: str = "paraformer"
    # ASR 推理线程池大小（CPU 密集，与 GPU 图像推理线程池隔离）
    VOICE_ASR_INFERENCE_THREADS: int = Field(default=2, gt=0)
    # 单次录音最大时长（秒）
    VOICE_ASR_MAX_DURATION: int = Field(default=60, gt=0)
    # 流式 ASR 最大并发会话数与 WebSocket 空闲超时（秒）
    VOICE_ASR_MAX_CONCURRENT_SESSIONS: int = Field(default=50, gt=0)
    VOICE_ASR_WS_IDLE_TIMEOUT: int = Field(default=60, gt=0)

    # 本地 TTS（Piper 引擎进程内运行，CPU 推理，模型自动从 hf-mirror 下载缓存）
    VOICE_TTS_VOICE_ID: str = "huayan"
    # TTS 推理线程池大小（CPU 密集，与 ASR/GPU 图像推理线程池隔离）
    VOICE_TTS_INFERENCE_THREADS: int = Field(default=2, gt=0)
    # 模型文件路径（默认 models/piper/{音色}.onnx，可经环境变量覆盖）
    VOICE_TTS_MODEL_PATH: str = ""
    # TTS 缓存：TTL（秒）/ 单用户最大缓存条数（LRU 淘汰）/ 音频 AES 加密密钥（环境变量注入）
    VOICE_TTS_CACHE_TTL: int = Field(default=86400, gt=0)
    VOICE_TTS_MAX_CACHE_PER_USER: int = Field(default=100, gt=0)
    VOICE_TTS_CACHE_ENCRYPTION_KEY: str = ""
    # 单次合成文本上限（字符）
    VOICE_TTS_MAX_TEXT_LENGTH: int = Field(default=10000, gt=0)

    # 语音计费单价（过渡配置：需求规格 §8 待确认事项 5 明确最终由 AI 计费管理模块统一定价）
    VOICE_ASR_CREDITS_PER_SECOND: float = Field(default=1.0, gt=0)
    VOICE_TTS_CREDITS_PER_CHAR: float = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def _apply_derived_credentials(self):
        """加载 .env 后统一复用凭证"""
        if not self.DEFAULT_PASSWORD:
            self.DEFAULT_PASSWORD = self.DEHAZE_PASSWORD
        if not self.XXLJOB_ACCESS_TOKEN:
            self.XXLJOB_ACCESS_TOKEN = self.DEHAZE_PASSWORD
        if not self.ES_PASSWORD:
            self.ES_PASSWORD = self.DEHAZE_PASSWORD
        return self


class DevelopmentSettings(Settings):
    """开发环境配置"""

    DEBUG: bool = True
    REDIS_DB: int = 0
    # HTTP + Vite 代理前缀 /py-api，需关闭 Secure 并用 /
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_PATH: str = "/"
    XXLJOB_ENABLED: bool = True
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_USER: str = "root"
    LOG_ARCHIVE_ON_STARTUP: bool = True
    # 开发环境调高限流上限，避免集成测试并行执行时触发限流
    RATE_LIMIT_MAX_REQUESTS: int = 10000
    COUPON_RECEIVE_RATE_LIMIT: int = 10000


class TestingSettings(Settings):
    """测试环境配置"""

    DEBUG: bool = True
    DB_NAME: str = "dehaze_test"
    RATE_LIMIT_MAX_REQUESTS: int = 10000
    COUPON_RECEIVE_RATE_LIMIT: int = 10000


class ProductionSettings(Settings):
    """生产环境配置"""

    XXLJOB_ENABLED: bool = True
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_USER: str = "root"
    SQL_LOG_LEVEL: Literal["INFO", "WARNING", "ERROR"] = "WARNING"
    LOG_FORMAT_JSON: bool = True
    PROMETHEUS_MULTIPROC_DIR: str = "/tmp/prometheus_multiproc"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.DEHAZE_PASSWORD:
            raise ValueError("生产环境必须设置 DEHAZE_PASSWORD 环境变量")
        # 禁止使用 localhost 白名单
        localhost_origins = any("localhost" in o or "127.0.0.1" in o for o in self.CORS_ORIGINS)
        if localhost_origins:
            raise ValueError(
                "生产环境 CORS_ORIGINS 禁止包含 localhost/127.0.0.1，请配置正式域名（逗号分隔）"
            )


_settings_map = {
    "development": DevelopmentSettings,
    "testing": TestingSettings,
    "production": ProductionSettings,
}


@lru_cache
def get_settings() -> Settings:
    """获取配置实例（缓存），用于打破循环导入的延迟入口"""
    env = os.getenv("APP_ENV", "development")
    settings_class = _settings_map.get(env)
    if settings_class is None:
        raise ValueError(f"未知的 APP_ENV={env!r}，可选值: {', '.join(_settings_map)}")
    return settings_class()


# 模块级单例：全局直接 from app.config import settings
settings = get_settings()

# 传播 PROMETHEUS_MULTIPROC_DIR 到 OS 环境变量
# prometheus_client 在导入时检查此环境变量来决定是否启用多进程模式，
# 必须在任何 prometheus_client 导入（如 starlette_exporter）之前完成
if settings.PROMETHEUS_MULTIPROC_DIR:
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings.PROMETHEUS_MULTIPROC_DIR
    os.makedirs(settings.PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
