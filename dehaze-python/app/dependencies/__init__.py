"""
FastAPI 依赖注入模块
"""

from app.database import get_db
from app.dependencies.auth import (
    get_current_user,
    get_current_user_optional,
    oauth2_scheme,
    UserContext,
)
from app.dependencies.redis import get_redis

__all__ = [
    "oauth2_scheme",
    "get_current_user",
    "get_current_user_optional",
    "UserContext",
    "get_db",
    "get_redis",
]
