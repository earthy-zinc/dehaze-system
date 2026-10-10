"""服务配置：读仓库根 .env，环境变量优先。

服务自身不查库，配置全部来自环境，可无状态重启与水平扩容。
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_service_root(start: Path) -> Path:
    """向上查找服务根：以 pyproject.toml 为标记，目录重构不会漂移。"""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"从 {start} 向上未找到 pyproject.toml，无法定位服务根")


SERVICE_ROOT = _find_service_root(Path(__file__).resolve().parent)
# 仓库根 = dehaze-service/dehaze-modal 的上两级；.env 与 models/ 均位于此处
PROJECT_ROOT = SERVICE_ROOT.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 应用与环境 =====
    APP_NAME: str = "Dehaze Modal"
    # 日志 fields.service（日志架构设计 §3.5 登记为 modal）
    SERVICE_NAME: str = "modal"
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8994, gt=0, le=65535)

    # ===== 服务间鉴权 =====
    # 部署级静态密钥（代表服务身份），与用户态凭证无关；默认值保证本地开发零摩擦
    INTERNAL_SERVICE_KEY: str = "local-no-key"

    # ===== 推理 =====
    # 线程池容量按单卡显存配置，超限请求排队不丢弃
    INFERENCE_THREAD_POOL_SIZE: int = Field(default=2, gt=0)
    INFERENCE_TIMEOUT_SECONDS: float = Field(default=30, gt=0)

    # ===== 权重来源 =====
    # 本地缓存优先，未命中从 nginx 静态服务（/models）下载
    MODEL_CACHE_DIR: str = str(PROJECT_ROOT / "models")
    NGINX_STATIC_HOST: str = "127.0.0.1"
    NGINX_STATIC_PORT: int = Field(default=19000, gt=0, le=65535)

    @property
    def MODEL_BASE_URL(self) -> str:
        """nginx 静态服务托管的算法权重根地址（/models 子路径）"""
        return f"http://{self.NGINX_STATIC_HOST}:{self.NGINX_STATIC_PORT}/models"

    # ===== GPU 监控 =====
    GPU_METRICS_INTERVAL_SECONDS: float = Field(default=5, gt=0)

    # ===== 日志 =====
    LOG_DIR: str = str(SERVICE_ROOT / "logs")
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = Field(default=100 * 1024 * 1024, gt=0)
    LOG_RETENTION_DAYS: int = Field(default=30, ge=0)


settings = Settings()
