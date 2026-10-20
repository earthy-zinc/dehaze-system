"""服务配置（pydantic-settings，读仓库根 .env，环境变量优先）。

本服务无数据库/缓存/存储依赖：配置只覆盖监听地址、模型缓存目录、引擎设备与线程数、
服务间密钥与日志。模型权重来源（ASR model_id、TTS onnx）由编排侧随请求下发，不在此配置。
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_service_root(start: Path) -> Path:
    """向上探测本服务根：以 pyproject.toml 为标记，不依赖文件相对层数，目录重构不会漂移"""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"从 {start} 向上未找到 pyproject.toml，无法定位 clarity-voice 服务根")


SERVICE_ROOT = _find_service_root(Path(__file__).resolve().parent)
# 仓库根（clarity-vision）：.env 与共享的 models/ 均位于此层
REPO_ROOT = SERVICE_ROOT.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== 服务标识 =====
    APP_NAME: str = "Dehaze Voice"
    # 日志 fields.service（日志架构设计 §3.5 登记为 voice）
    SERVICE_NAME: str = "voice"

    # ===== 监听（python -m app.main）=====
    HOST: str = "0.0.0.0"
    PORT: int = Field(default=8993, gt=0, le=65535)

    # ===== 服务间鉴权 =====
    # 编排侧以 Authorization: Bearer 携带；密钥未配置时全部请求 401（fail closed）
    INTERNAL_SERVICE_KEY: str = "local-no-key"

    # ===== 模型与引擎 =====
    # 模型持久化目录：ASR 走 ModelScope 缓存、TTS 走 piper 子目录，重启不重复下载
    MODEL_CACHE_DIR: str = str(REPO_ROOT / "models")
    # TTS 权重目录（留空则取 ${MODEL_CACHE_DIR}/piper）
    TTS_MODEL_DIR: str = ""
    # ASR：auto 时按 torch CUDA 可用性选择，也可显式指定 cpu / cuda:0
    ASR_DEVICE: str = "auto"
    ASR_INFERENCE_THREADS: int = Field(default=2, gt=0)
    # 流式会话无输入自动收尾（编排侧三个超时归其自有，这里只做兜底）
    ASR_STREAM_IDLE_TIMEOUT: float = Field(default=30.0, gt=0)

    TTS_INFERENCE_THREADS: int = Field(default=2, gt=0)
    # TTS 按字符数自限（TTS 无显式超时，服务侧按字符数自限）
    TTS_MAX_TEXT_CHARS: int = Field(default=10000, gt=0)

    # ===== 日志 =====
    LOG_DIR: str = str(SERVICE_ROOT / "logs")
    LOG_LEVEL: str = "INFO"
    LOG_ENABLE_CONSOLE: bool = True
    LOG_ENABLE_FILE: bool = True
    LOG_CONSOLE_JSON: bool = False
    LOG_RETENTION_DAYS: int = Field(default=30, ge=0)
    LOG_MAX_BYTES: int = Field(default=100 * 1024 * 1024, gt=0)

    @property
    def MODELSCOPE_CACHE_DIR(self) -> Path:
        """ASR 权重缓存根（ModelScope 约定的 hub/{命名空间}/{模型} 布局）"""
        return Path(self.MODEL_CACHE_DIR) / "modelscope"


settings = Settings()

# FunASR/ModelScope 的权重下载目录由环境变量决定，必须在本进程加载引擎前落地；
# setdefault 保证运维在 .env 之外显式指定的 MODELSCOPE_CACHE 优先
os.environ.setdefault("MODELSCOPE_CACHE", str(settings.MODELSCOPE_CACHE_DIR))
