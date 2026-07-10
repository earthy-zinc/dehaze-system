"""
Dehaze 应用包

FastAPI 入口: from app.main import app
"""
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
