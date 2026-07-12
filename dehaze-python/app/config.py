from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 所有环境都需要设置密钥
        if not self.SECRET_KEY:
            raise ValueError(
                "必须设置 SECRET_KEY 环境变量。"
                "开发环境可使用: export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
            )
        if not self.JWT_SECRET_KEY:
            raise ValueError(
                "必须设置 JWT_SECRET_KEY 环境变量。"
                "开发环境可使用: export JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
            )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用基础
    APP_NAME: str = "Dehaze API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # JWT 配置 - 必须通过环境变量设置，无安全默认值
    SECRET_KEY: str = Field(default="")  # 必须设置，启动时会校验
    JWT_SECRET_KEY: str = Field(default="")  # 必须设置，启动时会校验
    JWT_ACCESS_TOKEN_EXPIRES: int = 7200  # 访问令牌过期时间（秒），默认 2 小时
    JWT_REFRESH_TOKEN_EXPIRES: int = 604800  # 刷新令牌过期时间（秒），默认 7 天

    # 验证码配置
    CAPTCHA_LENGTH: int = 4  # 验证码字符数
    CAPTCHA_WIDTH: int = 120  # 验证码图片宽度
    CAPTCHA_HEIGHT: int = 40  # 验证码图片高度
    CAPTCHA_FONT_SIZE: int = 24  # 验证码字体大小
    CAPTCHA_NOISE_LINES: int = 5  # 干扰线数量
    CAPTCHA_EXPIRES: int = 300  # 验证码过期时间（秒），默认 5 分钟

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

    @property
    def REDIS_URL(self) -> str:
        if self.DEHAZE_PASSWORD:
            return f"redis://:{self.DEHAZE_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def REDIS_PASSWORD(self) -> Optional[str]:
        return self.DEHAZE_PASSWORD if self.DEHAZE_PASSWORD else None

    # MongoDB 配置
    MONGO_URI: str = "mongodb://127.0.0.1:27017/"

    # MinIO 配置（MinIO 移到 9100，9000 端口由 nginx-dataset 占用）
    MINIO_ENDPOINT: str = "127.0.0.1:9100"
    MINIO_ACCESS_KEY: str = ""  # 必须通过环境变量设置
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
    # 文件访问基础 URL（用于拼接文件访问地址）
    FILE_BASE_URL: str = ""
    # 本地存储路径（FILE_STORAGE_TYPE=local 时生效）
    LOCAL_STORAGE_PATH: str = "/data/files"
    # 孤儿文件清理阈值（小时）
    FILE_ORPHAN_CLEANUP_HOURS: int = 48
    # 临时文件清理阈值（小时）
    FILE_TEMP_CLEANUP_HOURS: int = 24

    # 文件存储
    BASE_URL: str = "http://localhost:8989/api/v1/files"
    DATASET_PATH: str = "/mnt/d/DeepLearning/dataset"

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
    def RABBITMQ_URL(self) -> str:
        password = self.DEHAZE_PASSWORD or self.RABBITMQ_USER
        return f"amqp://{self.RABBITMQ_USER}:{password}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/"

    # 任务并发限制
    TASK_MAX_CONCURRENT_PER_USER: int = 5  # 同用户同类型最大并发数

    # Prometheus 监控配置
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_GPU_COLLECT_INTERVAL: int = 5  # GPU 指标采集间隔（秒）

    # 优雅关闭配置
    GRACEFUL_SHUTDOWN_TIMEOUT: int = 30  # 等待任务完成超时（秒）
    GRACEFUL_SHUTDOWN_CANCEL_ON_TIMEOUT: bool = True  # 超时后是否取消任务

    # 日志配置
    LOG_LEVEL: str = "INFO"  # 日志级别: DEBUG/INFO/WARNING/ERROR/CRITICAL
    # 文本格式日志模板
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s --- [%(thread)d] %(name)s : %(message)s"
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
    DEFAULT_PASSWORD: str = "123456"  # 新用户默认密码
    PASSWORD_MIN_LENGTH: int = 8  # 密码最小长度
    PASSWORD_REQUIRE_COMPLEXITY: bool = True  # 是否要求密码复杂度（至少包含字母和数字）

    # 部门管理配置
    DEPT_MAX_DEPTH: int = 5  # 部门层级深度限制


class DevelopmentSettings(Settings):
    """开发环境配置"""

    DEBUG: bool = True
    DATABASE_ECHO: bool = True

    # 数据库配置
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306

    # Redis 配置
    REDIS_HOST: str = "127.0.0.1"
    REDIS_DB: int = 3

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
    MONGO_URI: str = "mongodb://192.168.31.3:27017/"
    MINIO_ENDPOINT: str = "192.168.31.3:9100"
    BASE_URL: str = "http://dehaze-python/api/v1/files"
    DATASET_PATH: str = "/app/dataset"

    # XXL-Job 配置（生产环境启用）
    XXLJOB_ENABLED: bool = True
    XXLJOB_ADMIN_URL: str = "http://192.168.31.3:8080/xxl-job-admin/api/"

    # RabbitMQ 配置（生产环境启用）
    RABBITMQ_ENABLED: bool = True
    RABBITMQ_HOST: str = "192.168.31.3"
    RABBITMQ_USER: str = "root"

    # 日志配置（生产环境启用 JSON 格式）
    LOG_FORMAT_JSON: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 生产环境强制校验关键配置
        if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
            raise ValueError("生产环境必须设置 SECRET_KEY 环境变量且长度 >= 32")
        if not self.JWT_SECRET_KEY or len(self.JWT_SECRET_KEY) < 32:
            raise ValueError("生产环境必须设置 JWT_SECRET_KEY 环境变量且长度 >= 32")
        if not self.DEHAZE_PASSWORD:
            raise ValueError("生产环境必须设置 DEHAZE_PASSWORD 环境变量")


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
    return settings_class()


settings = get_settings()
