from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _find_python_project_root(start: Path) -> Path:
    """向上查找 Python 项目根：以 pyproject.toml 为标记（Python 包必然存在），
    不依赖文件相对层数，目录重构不会漂移。"""
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise RuntimeError(f"从 {start} 向上未找到 pyproject.toml，无法定位 Python 项目根")


# dehaze-python 根（pyproject.toml 标记定位）
PYTHON_PROJECT_ROOT = _find_python_project_root(Path(__file__).resolve().parent)
# 仓库根（dehaze-system）= dehaze-python 的上级；.env、models、config/sql 位于此处。
# 集中为单一常量，其余路径（env_file、模型缓存等）以此为准，避免多处按 __file__ 重复推算
PROJECT_ROOT = PYTHON_PROJECT_ROOT.parent


class Settings(BaseSettings):
    """应用配置（多环境，通过 APP_ENV 切换）"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
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

    # ===== MySQL =====
    MYSQL_HOST: str = Field(default="127.0.0.1")
    MYSQL_PORT: int = Field(default=3306, gt=0, le=65535)
    # 环境身份类变量（库 / bucket / vhost / Redis db / ES 索引前缀）一律无默认值：
    # 缺失即启动失败，避免静默落到生产预留命名空间
    MYSQL_DATABASE: str
    MYSQL_USERNAME: str = "root"
    MYSQL_PASSWORD: str = Field(default="")
    DATABASE_POOL_SIZE: int = Field(default=10, gt=0)
    DATABASE_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DATABASE_POOL_RECYCLE: int = Field(default=3600, gt=0)
    # SQL 审计日志级别：INFO 记录全部 SQL（开发/测试），WARNING 仅记慢查询与错误（生产）
    SQL_LOG_LEVEL: Literal["INFO", "WARNING", "ERROR"] = "INFO"
    # 慢查询阈值（毫秒），超过则额外输出 WARNING 级 SLOW_SQL
    SQL_SLOW_THRESHOLD_MS: int = Field(default=500, gt=0)

    @property
    def DATABASE_URL(self) -> str:
        # 时区固化 Asia/Shanghai：连接后 SET time_zone='+08:00'（aiomysql init_command），
        # 不依赖服务器 SYSTEM 时区，保证 NOW()/CURRENT_TIMESTAMP/date_format 与应用写入时间一致
        return (
            f"mysql+aiomysql://{self.MYSQL_USERNAME}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            "?charset=utf8mb4&init_command=SET%20time_zone%20%3D%20%27%2B08%3A00%27"
        )

    # ===== Redis =====
    REDIS_HOST: str = Field(default="127.0.0.1")
    REDIS_PASSWORD: str = Field(default="")
    REDIS_PORT: int = Field(default=6379, gt=0, le=65535)
    REDIS_DATABASE: int = Field(ge=0)
    REDIS_MAX_CONNECTIONS: int = Field(default=100, gt=0)
    REDIS_SOCKET_TIMEOUT: float = Field(default=5.0, gt=0)
    REDIS_SOCKET_CONNECT_TIMEOUT: float = Field(default=5.0, gt=0)
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_HEALTH_CHECK_INTERVAL: int = Field(default=30, gt=0)

    @property
    def REDIS_URL(self) -> str:
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DATABASE}"

    # ===== MongoDB（审计日志） =====
    MONGODB_HOST: str = Field(default="127.0.0.1")
    MONGODB_PORT: int = Field(default=27017, gt=0, le=65535)
    MONGODB_DATABASE: str
    MONGODB_USERNAME: str = "root"
    MONGODB_PASSWORD: str = Field(default="")

    @property
    def MONGO_URI(self) -> str:
        return f"mongodb://{self.MONGODB_USERNAME}:{self.MONGODB_PASSWORD}@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DATABASE}?authSource=admin"

    # ===== 文件与模型存储 =====
    MINIO_HOST: str = Field(default="127.0.0.1")
    MINIO_PORT: int = Field(default=9110, gt=0, le=65535)
    MINIO_ACCESS_KEY: str = "admin"
    MINIO_SECRET_KEY: str = Field(default="")
    MINIO_SECURE: bool = False
    MINIO_BUCKET: str
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024, gt=0)
    # 上传/删除文件使用的默认存储后端（minio/local/nginx-static）
    FILE_STORAGE_TYPE: Literal["minio", "local", "nginx-static"] = "minio"
    LOCAL_STORAGE_PATH: str = str(PROJECT_ROOT / "data" / "upload")
    FILE_TEMP_CLEANUP_HOURS: int = Field(default=24, gt=0)
    # 孤儿文件保留阈值（小时）：物理文件存在但无元数据引用且超过该时长才清理
    ORPHAN_FILE_RETENTION_HOURS: int = Field(default=48, gt=0)
    TEMP_DIR: str = ""

    @property
    def MINIO_ENDPOINT(self) -> str:
        """MinIO 服务端点（host:port，不含 scheme，供 MinIO SDK 使用）"""
        return f"{self.MINIO_HOST}:{self.MINIO_PORT}"

    # Nginx 静态文件服务（datasets 数据集 / models 算法权重托管）
    NGINX_STATIC_HOST: str = Field(default="127.0.0.1")
    NGINX_STATIC_PORT: int = Field(default=9000, gt=0, le=65535)

    @property
    def MODEL_BASE_URL(self) -> str:
        return f"http://{self.NGINX_STATIC_HOST}:{self.NGINX_STATIC_PORT}/models"

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
            "minio": f"http://{self.MINIO_ENDPOINT}/{self.MINIO_BUCKET}",
            "local": "http://127.0.0.1:8989/api/v1/files/download",
            "nginx-static": f"http://{self.NGINX_STATIC_HOST}:{self.NGINX_STATIC_PORT}",
        }

    @property
    def TEMP_DIR_RESOLVED(self) -> str:
        """未设置 TEMP_DIR 时使用系统临时目录"""
        import tempfile

        return self.TEMP_DIR if self.TEMP_DIR else tempfile.gettempdir()

    # ===== RabbitMQ =====
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_HOST: str = Field(default="127.0.0.1")
    RABBITMQ_PORT: int = Field(default=5672, gt=0, le=65535)
    RABBITMQ_USERNAME: str = "guest"
    RABBITMQ_PASSWORD: str = Field(default="")
    # vhost（URL 路径部分）：%2F 为默认 vhost（生产），dev / pre 环境各用独立 vhost
    RABBITMQ_VHOST: str
    RABBITMQ_EXCHANGE: str = "dehaze.tasks"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_RECONNECT_MAX_RETRIES: int = Field(default=0, ge=0)  # 0 表示无限重试
    RABBITMQ_RECONNECT_INITIAL_INTERVAL: float = Field(default=1.0, gt=0)
    RABBITMQ_RECONNECT_MAX_INTERVAL: float = Field(default=30.0, gt=0)
    RABBITMQ_PREFETCH_COUNT: int = Field(default=2, gt=0)
    RABBITMQ_RETRY_DELAYS: list[int] = [5000, 30000, 300000]  # 分级重试延迟（ms）: 5s/30s/5min

    @property
    def RABBITMQ_URL(self) -> str:
        return (
            f"amqp://{self.RABBITMQ_USERNAME}:{self.RABBITMQ_PASSWORD}"
            f"@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/{self.RABBITMQ_VHOST}"
        )

    # ===== XXL-Job 定时任务 =====
    XXLJOB_ENABLED: bool = False
    XXLJOB_HOST: str = Field(default="127.0.0.1")
    XXLJOB_PORT: int = Field(default=14980, gt=0, le=65535)
    XXLJOB_ACCESS_TOKEN: str = Field(default="")
    XXLJOB_EXECUTOR_APP_NAME: str = "xxl-job-executor-dehaze-python"
    XXLJOB_EXECUTOR_PORT: int = Field(default=9998, gt=0, le=65535)
    # 执行器注册地址（admin 回调用），留空由 pyxxl 取首个网卡 IP；
    # admin 在容器时由 run.py 注入 host.docker.internal
    XXLJOB_EXECUTOR_IP: str = ""
    XXLJOB_TASK_LOG_DIR: str = str(PYTHON_PROJECT_ROOT / "logs" / "xxljob-tasks")
    XXLJOB_PID_FILE: str = str(PYTHON_PROJECT_ROOT / "logs" / "pyxxl.pid")

    @property
    def XXLJOB_ADMIN_URL(self) -> str:
        return f"http://{self.XXLJOB_HOST}:{self.XXLJOB_PORT}/xxl-job-admin/api/"

    # ===== 日志 =====
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: str = (
        "%(asctime)s - %(levelname)s [%(trace_id)s] --- [%(thread)d] %(name)s : %(message)s"
    )
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_DIR: str = str(PYTHON_PROJECT_ROOT / "logs")
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
    # 设置后 /metrics 端点通过 MultiProcessCollector 聚合所有 Worker 指标
    PROMETHEUS_MULTIPROC_DIR: str = ""

    # ===== 安全与认证 =====
    # 验证码
    CAPTCHA_LENGTH: int = Field(default=4, ge=4)
    CAPTCHA_WIDTH: int = Field(default=120, gt=0)
    CAPTCHA_HEIGHT: int = Field(default=40, gt=0)
    CAPTCHA_FONT_SIZE: int = Field(default=24, gt=0)
    CAPTCHA_NOISE_LINES: int = Field(default=5, ge=0)
    CAPTCHA_EXPIRES: int = Field(default=300, gt=0)

    # 注册开关（用户注册设计.md §9：运营可在需要时临时关闭注册）
    REGISTER_ENABLED: bool = True

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

    # 密码策略（8-20 位含字母和数字）
    PASSWORD_MIN_LENGTH: int = Field(default=8, gt=0)
    PASSWORD_MAX_LENGTH: int = Field(default=20, gt=0)
    PASSWORD_REQUIRE_COMPLEXITY: bool = True

    # CORS 跨域来源白名单（三端同名环境变量 CORS_ALLOWED_ORIGINS：Java 为 cors.allowed-origins、
    # Go 为 cors.allowed-origins，换环境即换 .env）
    # 默认值为开发端口白名单（端口规范：5173 React / 5174 Vue / 5175 Taro / 5176 uniapp /
    # 5177 Flutter Web / 5183 React Electron / 5184 Vue Electron / 8081 RN Metro）
    CORS_ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = [
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

    @field_validator("CORS_ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_allowed_origins(cls, v):
        """支持环境变量传入逗号分隔字符串或 JSON 数组"""
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # 优雅关闭等待时长（秒）
    GRACEFUL_SHUTDOWN_TIMEOUT: int = Field(default=30, gt=0)

    # ===== 业务配置 =====
    # 新用户初始密码（三端统一，由根目录 .env 显式提供）
    DEFAULT_PASSWORD: str = Field(default="")

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
    # 单条消息长度按所选模型上下文预算校验（见 service/ai/service/model_defaults）
    # 流式输出超时（秒）
    AI_MESSAGE_STREAM_TIMEOUT: int = 120
    # SSE 心跳间隔（秒）
    AI_MESSAGE_HEARTBEAT_INTERVAL: int = 15
    # 默认模型与模型清单均不在配置项：模型由 sys_ai_provider/sys_ai_model 注册表驱动，
    # 平台默认模型由 sys_dict（ai_model.default_model）配置，未配置则报错
    # 内置本地推理服务（dehaze-service/dehaze-llm，OpenAI 兼容：chat / embeddings / rerank）。
    # local provider 的 api_base_url 由该地址派生；推理参数（模型路径/线程数/GPU 层数）
    # 全部下沉服务侧，编排侧只保留地址、就绪要求与上下文长度
    LOCAL_LLM_BASE_URL: str = "http://127.0.0.1:8992"
    # 本地推理是否为必选依赖：默认 False（非必选）——未启动/未就绪不阻断整体就绪，
    # /ready 仍展示其状态；本地 LLM 作为必选依赖的部署显式置 True
    LOCAL_LLM_REQUIRED: bool = False
    # 仅用于播种 rerank 模型行的 max_context_tokens，须与服务侧 CTX_SIZE 一致
    LOCAL_LLM_CTX_SIZE: int = 16384
    # 服务间调用密钥（部署级静态密钥，代表服务身份，与用户 API Key 不混用）：
    # 编排侧发起调用经 Authorization: Bearer 携带，推理服务校验入站请求；
    # local provider 的占位 Key 明文即取该值，使供应商链路与鉴权同源
    INTERNAL_SERVICE_KEY: str = "local-no-key"
    # 语音引擎（dehaze-voice）地址：语音引擎统一按第三方供应商注册于库内，
    # 该地址仅用于播种内置引擎供应商行的 api_base_url（对齐 LOCAL_LLM_BASE_URL 的播种口径）
    VOICE_ENGINE_BASE_URL: str = "http://127.0.0.1:8993"
    # 内置语音引擎是否在本部署启用：置 False 表示 ASR/TTS 全由第三方引擎提供
    # （`python -m app.cli init-builtin` 不初始化 local-voice 并禁用已有行；运行期不写库）
    VOICE_LOCAL_ENABLED: bool = True
    # 图像推理引擎地址（dehaze-service/dehaze-modal）
    MODAL_ENGINE_BASE_URL: str = "http://127.0.0.1:8994"
    # 速度档位 P95 延迟阈值（毫秒）：P95 < fast 为快；< medium 为中；其余为慢
    AI_SPEED_TIER_FAST_P95_MS: int = 2000
    AI_SPEED_TIER_MEDIUM_P95_MS: int = 8000
    # 供应商 API Key 加密密钥（AES-256，环境变量注入）
    AI_PROVIDER_KEY_ENCRYPTION_KEY: str = ""
    # MCP 能力网关
    MCP_GATEWAY_URL: str = "http://127.0.0.1:8082/mcp"
    # 网关共享密钥（Authorization: Bearer 请求头，与根 .env 同源）；未配置时拒绝连接网关
    MCP_GATEWAY_KEY: str = ""
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
    # 会员卡收入归属 AI 的分摊比例（0~1）：仅用于 AI 参考毛利核算，不影响整体毛利官方口径
    AI_BILLING_MEMBERSHIP_AI_RATIO: float = 0.3
    # 成本核算开关（关闭后不执行异步成本回填）
    AI_BILLING_COST_CALC_ENABLE: bool = True
    # 月结账单 Redis 缓存 TTL（秒，90 天）
    AI_BILLING_BILL_CACHE_TTL: int = 90 * 24 * 3600

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

    # ===== Elasticsearch（记忆向量检索 / 会话全文检索，必选基础设施，
    # docker-compose 统一部署） =====
    ES_URL: str = "http://localhost:9200"
    ES_API_KEY: str = ""  # 可选；与用户名密码二选一
    ES_USERNAME: str = "elastic"  # docker-compose 启用 xpack security，elastic 超级用户
    ES_PASSWORD: str = Field(default="")
    # 索引名前缀（多环境隔离）：空 = 生产；dev / pre 分别为 dev_ / pre_
    ES_INDEX_PREFIX: str

    # ===== AI 知识库 =====
    # 分块/检索的业务默认值（分块大小与重叠、Top-K、相似度阈值）以库内
    # sys_knowledge_base 对应列为准；RRF 常量、索引批量与重试次数为实现常量
    # （见 app/core/constants.py），不在此配置
    KB_SEARCH_TIMEOUT_MS: int = Field(
        default=10000,
        gt=0,
        description=(
            "检索整体超时(ms)，超时降级无知识回复（对话注入链路）。该预算覆盖完整检索链路"
            "（查询向量化 + ES 检索/融合 + 节内容回填 + rerank），非仅 ES 检索："
            "provider=local 时查询向量化走与对话推理共用的 CPU 子进程，实测单次 0.4~3s"
            "（冷启动/被对话占用时更久），叠加 ES 检索 ~0.4s，2s 预算会把健康检索误判为超时"
            "并静默降级为空结果。默认值按实测链路留 5x 余量"
        ),
    )
    KB_SEARCH_QUERY_TIMEOUT_MS: int = Field(
        default=30000,
        gt=0,
        description=(
            "显式检索接口（/kb/search、/kb/{id}/retrieve/test、召回测试集执行）的整体超时预算(ms)。"
            "这些链路由用户/管理员主动发起且正等待结果，返回空结果会被误读为「库内无匹配」"
            "（检索测试工具正是如此误报），故预算需覆盖本地 CPU 向量化的尾延迟；"
            "对话注入链路仍用 KB_SEARCH_TIMEOUT_MS，保证对话不被慢检索阻塞"
        ),
    )
    # 索引与异步处理
    KB_INDEX_BATCH_SIZE: int = Field(default=100, gt=0, description="ES 批量索引条数")
    KB_ASYNC_MAX_RETRY: int = Field(default=3, ge=0, description="异步处理最大重试次数")
    KB_INDEX_WARNING_THRESHOLD: int = Field(
        default=1 * 1024 * 1024 * 1024, gt=0, description="索引大小阈值告警(字节)，默认 1GB"
    )

    # ===== 语音交互 =====
    # 识别与合成经统一模型路由（sys_ai_model + sys_ai_provider，默认模型取
    # ai_model.default_model.asr/.tts）：引擎推理工件由服务端目录解析，应用侧不下发
    # 单次录音最大时长（秒）
    VOICE_ASR_MAX_DURATION: int = Field(default=60, gt=0)
    # 流式 ASR 最大并发会话数与 WebSocket 空闲超时（秒）
    VOICE_ASR_MAX_CONCURRENT_SESSIONS: int = Field(default=50, gt=0)
    VOICE_ASR_WS_IDLE_TIMEOUT: int = Field(default=60, gt=0)

    # TTS 默认音色（未指定 voice 时使用，须在引擎侧音色目录内）
    VOICE_TTS_VOICE_ID: str = "huayan"
    # TTS 缓存：TTL（秒）/ 单用户最大缓存条数（LRU 淘汰）/ 音频 AES 加密密钥（环境变量注入）
    VOICE_TTS_CACHE_TTL: int = Field(default=86400, gt=0)
    VOICE_TTS_MAX_CACHE_PER_USER: int = Field(default=100, gt=0)
    VOICE_TTS_CACHE_ENCRYPTION_KEY: str = ""
    # 单次合成文本上限（字符）
    VOICE_TTS_MAX_TEXT_LENGTH: int = Field(default=10000, gt=0)

    # 语音计费单价不在配置项：单价随 AI 计费统一管理（sys_ai_model_price，按 billing_unit 计量）


class DevelopmentSettings(Settings):
    """开发环境配置"""

    DEBUG: bool = True
    # HTTP + Vite 代理前缀 /py-api，需关闭 Secure 并用 /
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_PATH: str = "/"
    XXLJOB_ENABLED: bool = True
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_USERNAME: str = "root"
    LOG_ARCHIVE_ON_STARTUP: bool = True
    # 开发环境调高限流上限，避免集成测试并行执行时触发限流
    RATE_LIMIT_MAX_REQUESTS: int = 10000
    COUPON_RECEIVE_RATE_LIMIT: int = 10000


class TestingSettings(Settings):
    """测试环境配置"""

    DEBUG: bool = True
    MYSQL_DATABASE: str = "dehaze_test"
    # Redis 指向无人监听端口，作为第三层兜底（前两层：conftest 的 mock_redis
    # autouse fixture 中心 patch + 动态扫描顶层直接引用的模块；此端口仅防函数
    # 别名导入等前两层覆盖不到的路径）。勿改用端口 0：redis-py 把 0 当默认
    # 6379，实测会连上真实 Redis
    REDIS_PORT: int = 6390
    RATE_LIMIT_MAX_REQUESTS: int = 10000
    COUPON_RECEIVE_RATE_LIMIT: int = 10000


class ProductionSettings(Settings):
    """生产环境配置"""

    XXLJOB_ENABLED: bool = True
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_USERNAME: str = "root"
    SQL_LOG_LEVEL: Literal["INFO", "WARNING", "ERROR"] = "WARNING"
    LOG_FORMAT_JSON: bool = True
    PROMETHEUS_MULTIPROC_DIR: str = str(Path(tempfile.gettempdir()) / "prometheus_multiproc")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.MYSQL_PASSWORD:
            raise ValueError("生产环境必须设置 MYSQL_PASSWORD 环境变量")
        # 禁止使用 localhost 白名单
        localhost_origins = any(
            "localhost" in o or "127.0.0.1" in o for o in self.CORS_ALLOWED_ORIGINS
        )
        if localhost_origins:
            raise ValueError(
                "生产环境 CORS_ALLOWED_ORIGINS 禁止包含 localhost/127.0.0.1，"
                "请配置正式域名（逗号分隔）"
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
    # 必填字段（库名/桶名/vhost 等）由环境变量注入，类型检查器无法据实参推断，此处豁免
    return settings_class()  # pyright: ignore[reportCallIssue]


# 模块级单例：全局直接 from app.config import settings
settings = get_settings()

# 传播 PROMETHEUS_MULTIPROC_DIR 到 OS 环境变量
# prometheus_client 在导入时检查此环境变量来决定是否启用多进程模式，
# 必须在任何 prometheus_client 导入（如 starlette_exporter）之前完成
if settings.PROMETHEUS_MULTIPROC_DIR:
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = settings.PROMETHEUS_MULTIPROC_DIR
    Path(settings.PROMETHEUS_MULTIPROC_DIR).mkdir(parents=True, exist_ok=True)
