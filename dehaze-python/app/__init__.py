"""
Dehaze 应用包

FastAPI 入口: from app.main import app
"""
# 在任何子模块导入之前配置 CUDA 扩展编译环境（MSVC / ninja / 缓存目录）
from app import _bootstrap  # noqa: F401

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings
    from fastapi import FastAPI
    app: FastAPI
    settings: Settings

# 延迟导入，避免循环依赖
__all__ = ["app", "settings"]


def __getattr__(name):
    if name == "app":
        from app.main import app
        return app
    if name == "settings":
        from app.config import settings
        return settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
