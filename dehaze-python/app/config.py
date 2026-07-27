from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有环境都需要设置密钥
        if not self.JWT_SECRET_KEY:
            raise ValueError(
                "必须设置 JWT_SECRET_KEY 环境变量。"
                "开发环境可使用: export JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
            )

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用基础
    APP_NAME: str = "Dehaze API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # JWT 配置 - 必须通过环境变量设置，无安全默认值
    JWT_SECRET_KEY: str = Field(default="")  # 必须设置，启动时会校验
    JWT_ACCESS_TOKEN_EXPIRES: int = 7200  # 访问令牌过期时间（秒），默认 2 小时
    JWT_REFRESH_TOKEN_EXPIRES: int = 604800  # 刷新令牌过期时间（秒），默认 7 天

    # 验证码配置
    CAPTCHA_KEY_PREFIX: str = "captcha_code:"  # 验证码 Redis key 前缀，与 Java/Go 保持一致
    CAPTCHA_LENGTH: int = 4
    CAPTCHA_WIDTH: int = 120
    CAPTCHA_HEIGHT: int = 40
    CAPTCHA_FONT_SIZE: int = 24
    CAPTCHA_NOISE_LINES: int = 5
    CAPTCHA_EXPIRES: int = 300

    # Session Cookie 配置（与 Java/Go 保持一致）
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_PATH: str = "/api"

    # 共享密码（从 .env 加载）
    DEHAZE_PASSWORD: str = Field(default="")

    # 数据库配置
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_NAME: str = "dehaze"
    DB_USER: str = "root"

    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_ECHO: bool = False

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+aiomysql://{self.DB_USER}:{self.DEHAZE_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"

    # Redis 配置
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 20  # 连接池最大连接数
    REDIS_SOCKET_TIMEOUT: float = 5.0  # 操作超时（秒）
    REDIS_SOCKET_CONNECT_TIMEOUT: float = 5.0  # 连接超时（秒）
    REDIS_RETRY_ON_TIMEOUT: bool = True  # 超时是否重试
    REDIS_HEALTH_CHECK_INTERVAL: int = 30  # 健康检查间隔（秒）

    # ===== 多级缓存防护配置 =====
    # L1 本地缓存（防热 key 击穿 Redis）
    CACHE_L1_ENABLED: bool = True
    CACHE_L1_MAXSIZE: int = 1000  # 最大缓存条目数
    CACHE_L1_TTL: int = 300  # L1 默认 TTL（秒），5 分钟
    # SingleFlight（防缓存击穿，热点 key 失效瞬间合并并发加载）
    CACHE_SINGLEFLIGHT_ENABLED: bool = True
    # 空值缓存（防缓存穿透）
    CACHE_NULL_ENABLED: bool = True
    CACHE_NULL_TTL: int = 60  # 空值缓存 TTL（秒）

    @property
    def REDIS_URL(self) -> str:
        if self.DEHAZE_PASSWORD:
            return f"redis://:{self.DEHAZE_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # MinIO 配置
    MINIO_ENDPOINT: str = "127.0.0.1:9110"
    MINIO_ACCESS_KEY: str = "admin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "dehaze"

    # 文件上传限制
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: set[str] = {
        # 图片
        "jpg", "jpeg", "png", "gif", "webp", "bmp", "svg",
        # 文档
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        # 压缩包
        "zip", "rar", "7z", "tar", "gz",
        # 其他
        "txt", "json", "xml", "csv",
    }

    @property
    def MINIO_SECRET_KEY(self) -> str:
        return self.DEHAZE_PASSWORD

    # 文件存储策略: minio / local
    FILE_STORAGE_TYPE: str = "minio"
    # 文件访问基础 URL（用于拼接文件访问地址）。
    # 留空时返回相对路径 /api/v1/files/download/...；配置后返回绝对 URL，与 Java file.baseUrl 风格一致。
    FILE_BASE_URL: str = ""
    # 本地存储路径（FILE_STORAGE_TYPE=local 时生效）
    LOCAL_STORAGE_PATH: str = "/data/files"
    # 孤儿文件清理阈值（小时）
    FILE_ORPHAN_CLEANUP_HOURS: int = 48
    # 临时文件清理阈值（小时）
    FILE_TEMP_CLEANUP_HOURS: int = 24

    # 模型权重文件存储配置
    # Nginx 静态服务（nginx-dataset 容器）下 /models 路径的基础 URL，
    # 算法权重文件通过 HTTP 下载到本地缓存后由 torch.load 加载，解除 Python 服务与 trained_model/ 目录的强耦合。
    # 算法权重访问 URL = {MODEL_BASE_URL}/{algorithm.path}，如 http://127.0.0.1:9000/models/AECR-Net/NH_train.pk
    MODEL_BASE_URL: str = "http://127.0.0.1:9000/models"
    # 本地缓存目录：首次下载后缓存到此目录，二次加载直接读缓存。
    # 默认指向项目内 trained_model/，本地开发时无需重复下载占用空间
    MODEL_CACHE_DIR: str = "./trained_model"
    # Nginx 不可用时是否降级到本地缓存（仅当本地已存在缓存文件时生效）
    MODEL_FALLBACK_TO_LOCAL: bool = True

    # 文件存储

    # 临时文件目录
    TEMP_DIR: str = ""

    @property
    def TEMP_DIR_RESOLVED(self) -> str:
        """解析临时目录，未设置时使用系统临时目录"""
        import tempfile
        return self.TEMP_DIR if self.TEMP_DIR else tempfile.gettempdir()

    # 设备配置
    DEVICE_ID: list[int] = [0]

    # XXL-Job 定时任务配置
    XXLJOB_ENABLED: bool = False
    XXLJOB_ADMIN_URL: str = "http://localhost:8080/xxl-job-admin/api/"
    XXLJOB_ACCESS_TOKEN: str = "default_token"
    XXLJOB_EXECUTOR_APP_NAME: str = "xxl-job-executor-dehaze-python"
    XXLJOB_EXECUTOR_HOST: str = "0.0.0.0"
    XXLJOB_EXECUTOR_PORT: int = 9999
    XXLJOB_EXECUTOR_LOG_PATH: str = "logs/pyxxl.log"  # 执行器自身运行日志
    XXLJOB_TASK_LOG_DIR: str = "logs/xxljob-tasks"  # 调度任务执行日志目录
    XXLJOB_PID_FILE: str = "logs/pyxxl.pid"  # 执行器子进程 PID 文件

    # RabbitMQ 配置
    RABBITMQ_ENABLED: bool = False
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_EXCHANGE: str = "dehaze.task"
    RABBITMQ_EXCHANGE_TYPE: str = "direct"
    RABBITMQ_ROUTING_KEY_PREFIX: str = "task"
    RABBITMQ_RECONNECT_MAX_RETRIES: int = 0  # 0 表示无限重试
    RABBITMQ_RECONNECT_INITIAL_INTERVAL: float = 1.0  # 首次重连间隔（秒）
    RABBITMQ_RECONNECT_MAX_INTERVAL: float = 30.0  # 退避上限（秒）
    RABBITMQ_PREFETCH_COUNT: int = 2  # 消费者预取数量
    RABBITMQ_RETRY_DELAYS: list[int] = [
        5000, 30000, 300000]  # 分级重试延迟（ms）: 5s/30s/5min

    @property
    def RABBITMQ_PASSWORD(self) -> str:
        """RabbitMQ 密码：复用 DEHAZE_PASSWORD 统一凭证，未设置时返回开发默认值 guest"""
        return self.DEHAZE_PASSWORD or "guest"

    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/%2F"

    # 任务并发限制
    TASK_MAX_CONCURRENT_PER_USER: int = 5  # 同用户同类型最大并发数

    # Prometheus 监控配置
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_GPU_COLLECT_INTERVAL: int = 5  # GPU 指标采集间隔（秒）
    # 多 Worker 模式下的指标聚合目录（PROMETHEUS_MULTIPROC_DIR）
    # 设置后 /metrics 端点将通过 MultiProcessCollector 聚合所有 Worker 的指标
    # 留空则使用单进程模式（仅返回当前 Worker 的指标）
    PROMETHEUS_MULTIPROC_DIR: str = ""

    # 优雅关闭配置
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30  # 等待任务完成超时（秒）
    GRACEFUL_SHUTDOWN_CANCEL_ON_TIMEOUT: bool = True  # 超时后是否取消任务

    # 日志配置
    LOG_LEVEL: str = "INFO"  # 日志级别: DEBUG/INFO/WARNING/ERROR/CRITICAL
    # 文本格式日志模板
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s [%(trace_id)s] --- [%(thread)d] %(name)s : %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"  # 日期格式
    LOG_DIR: str = "logs"  # 日志目录
    LOG_FILE: str = "dehaze-python.log"  # 日志文件名
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 单个日志文件最大字节数（10MB）
    LOG_BACKUP_COUNT: int = 5  # 保留的备份文件数量
    LOG_ENABLE_CONSOLE: bool = True  # 是否启用控制台输出
    LOG_ENABLE_FILE: bool = True  # 是否启用文件输出
    LOG_ROTATION_TYPE: str = "size"  # 轮转类型: size（基于大小）/ time（基于时间）
    LOG_FORMAT_JSON: bool = False  # 是否使用 JSON 结构化日志（生产环境推荐 True）

    # 用户管理配置
    DEFAULT_PASSWORD: str = "12345678"  # 新用户默认密码
    PASSWORD_MIN_LENGTH: int = 8  # 密码最小长度
    PASSWORD_REQUIRE_COMPLEXITY: bool = True  # 是否要求密码复杂度（至少包含字母和数字）

    # 部门管理配置
    DEPT_MAX_DEPTH: int = 5  # 部门层级深度限制

    # ===== CORS 跨域配置 =====
    # 开发环境白名单（生产环境通过 ProductionSettings 覆盖或 CORS_ORIGINS 环境变量配置）
    # 端口规范：5173 React / 5174 Vue / 5175 Taro / 5176 uniapp / 5177 Flutter Web / 5183 React Electron / 5184 Vue Electron / 8081 RN Metro
    CORS_ORIGINS: list[str] = [
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

    # ===== 安全防护配置（平台级） =====
    # IP 黑名单（自动封禁异常请求的 IP）
    IP_BLACKLIST_ENABLED: bool = True
    IP_BLACKLIST_THRESHOLD: int = 100  # 追踪窗口内异常请求次数阈值
    IP_BLACKLIST_DURATION: int = 3600  # 自动封禁时长（秒），默认 1 小时
    IP_BLACKLIST_TRACKING_WINDOW: int = 60  # 异常请求追踪窗口（秒）

    # ===== WebSocket 跨 Worker 配置 =====
    WS_REDIS_CHANNEL: str = "dehaze:ws:broadcast"  # Pub/Sub 频道名
    WS_ONLINE_KEY: str = "dehaze:ws:online_users"  # 在线用户 Redis sorted set key
    WS_HEARTBEAT_INTERVAL: int = 30  # 心跳间隔（秒）
    WS_ONLINE_TTL: int = 90  # 在线状态过期时间（秒），应 >= 3 倍心跳间隔

    # ===== TaskTracker 跨 Worker 配置 =====
    TASK_REDIS_KEY_PREFIX: str = "task:running"  # Redis 任务状态 key 前缀
    TASK_HEARTBEAT_INTERVAL: int = 30  # 任务心跳间隔（秒）
    TASK_REDIS_TTL: int = 3600  # Redis 任务状态 TTL（秒）


class DevelopmentSettings(Settings):
    """开发环境配置"""

    DEBUG: bool = True
    DATABASE_ECHO: bool = True

    # 数据库配置
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306

    # Redis 配置
    REDIS_HOST: str = "127.0.0.1"
    REDIS_DB: int = 0

    # Session Cookie：开发环境 HTTP + Vite 代理前缀 /py-api，需关闭 Secure 并用 /
    SESSION_COOKIE_SECURE: bool = False
    SESSION_COOKIE_PATH: str = "/"

    # 文件访问基础 URL（与 Java 端 file.baseUrl 一致，统一使用 127.0.0.1 避免 localhost DNS 解析开销）
    FILE_BASE_URL: str = "http://127.0.0.1:8989/api/v1/files/download"

    # XXL-Job 配置（Docker 中未运行 xxl-job-admin，关闭避免启动卡住）
    XXLJOB_ENABLED: bool = False
    XXLJOB_ADMIN_URL: str = "http://127.0.0.1:14980/xxl-job-admin/api/"

    # RabbitMQ 配置
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_HOST: str = "127.0.0.1"
    RABBITMQ_USER: str = "root"


class TestingSettings(Settings):
    """测试环境配置"""

    DEBUG: bool = True
    DB_NAME: str = "dehaze_test"


class ProductionSettings(Settings):
    """生产环境配置"""

    DB_HOST: str = "192.168.31.3"
    REDIS_HOST: str = "192.168.31.3"
    MINIO_ENDPOINT: str = "192.168.31.3:9110"

    # XXL-Job 配置（生产环境启用）
    XXLJOB_ENABLED: bool = True
    XXLJOB_ADMIN_URL: str = "http://192.168.31.3:8080/xxl-job-admin/api/"

    # RabbitMQ 配置（生产环境启用）
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_HOST: str = "192.168.31.3"
    RABBITMQ_USER: str = "root"

    # 日志配置（生产环境启用 JSON 格式）
    LOG_FORMAT_JSON: bool = True

    # Prometheus 多 Worker 指标聚合（生产环境强制启用）
    PROMETHEUS_MULTIPROC_DIR: str = "/tmp/prometheus_multiproc"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 生产环境强制校验关键配置
        if not self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
            raise ValueError("生产环境必须设置 JWT_SECRET_KEY 环境变量且长度 >= 32")
        if not self.DEHAZE_PASSWORD:
            raise ValueError("生产环境必须设置 DEHAZE_PASSWORD 环境变量")
        # 生产环境 CORS_ORIGINS 必须从环境变量配置（禁止使用 localhost 白名单）
        if not os.getenv("CORS_ORIGINS"):
            raise ValueError("生产环境必须设置 CORS_ORIGINS 环境变量（逗号分隔的域名白名单）")


_settings_map = {
    "development": DevelopmentSettings,
    "testing": TestingSettings,
    "production": ProductionSettings,
}


@lru_cache
def get_settings() -> Settings:
    """获取配置实例（缓存）"""
    env = os.getenv("APP_ENV", "development")
    settings_class = _settings_map.get(env, DevelopmentSettings)
    instance = settings_class()

    # 传播 PROMETHEUS_MULTIPROC_DIR 到 OS 环境变量
    # prometheus_client 在导入时检查此环境变量来决定是否启用多进程模式
    # 必须在任何 prometheus_client 导入（如 starlette_exporter）之前设置
    if instance.PROMETHEUS_MULTIPROC_DIR:
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = instance.PROMETHEUS_MULTIPROC_DIR
        os.makedirs(instance.PROMETHEUS_MULTIPROC_DIR, exist_ok=True)

    return instance


settings = get_settings()
