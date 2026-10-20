"""服务配置（pydantic-settings，读仓库根 .env，环境变量优先），不继承 clarity-python。

本服务无数据库/缓存/存储依赖：配置只覆盖监听地址、模型缓存与推理参数、服务间密钥与日志。
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_service_root(start: Path) -> Path:
    """向上探测本服务根：以 pyproject.toml 为标记，不依赖文件相对层数，目录重构不会漂移"""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"从 {start} 向上未找到 pyproject.toml，无法定位 clarity-llm 服务根")


SERVICE_ROOT = _find_service_root(Path(__file__).resolve().parent)
# 仓库根（clarity-vision）：.env 与 models/ 均位于此处，与编排侧共用同一份模型缓存
PROJECT_ROOT = SERVICE_ROOT.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 服务标识 =====
    APP_NAME: str = "Dehaze LLM"
    # 日志 fields.service（日志架构设计 §3.5 登记为 llm）
    SERVICE_NAME: str = "llm"

    # ===== 监听（python -m app.main）=====
    # 服务只绑内网：调用方是编排侧与 Prometheus，不直面 java/go 与前端
    HOST: str = "127.0.0.1"
    PORT: int = Field(default=8992, gt=0, le=65535)

    # ===== 服务间鉴权 =====
    # 入站鉴权密钥，与编排侧 INTERNAL_SERVICE_KEY 同源。部署级静态密钥（代表服务身份），
    # 与用户 API Key 不混用；默认值保证本地开发零摩擦，pre/prod 在 .env 覆盖
    INTERNAL_SERVICE_KEY: str = "local-no-key"

    # ===== 模型与推理 =====
    MODEL_CACHE_DIR: str = str(PROJECT_ROOT / "models")
    MODEL_PATH: str = ""  # 空 = 默认 models/Qwen3-0.6B-Q4_K_M.gguf（对话）
    EMBEDDING_MODEL_PATH: str = ""  # 空 = 默认 models/Qwen3-Embedding-0.6B-Q8_0.gguf（向量）
    RERANK_MODEL_PATH: str = ""  # 空 = 默认 models/Qwen3-Reranker-0.6B-q8_0.gguf（重排）
    # 推理参数：CTX_SIZE 由对话与重排共用（重排输入是 query + 文档节全文，小上下文会截断文档）
    CTX_SIZE: int = Field(default=16384, gt=0)
    THREADS: int = Field(default=0, ge=0)  # 0 = 自动（物理核数）
    NGPU_LAYERS: int | None = None  # None = 自动探测（CUDA 构建全量卸载，纯 CPU 构建 0）

    # ===== 日志 =====
    LOG_DIR: str = str(SERVICE_ROOT / "logs")
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = Field(default=100 * 1024 * 1024, gt=0)
    LOG_RETENTION_DAYS: int = Field(default=30, ge=0)


settings = Settings()
