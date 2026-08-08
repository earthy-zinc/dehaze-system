"""调试工具库：三端后端、Redis、MySQL 连接配置。

对齐 dehaze-sdk-js/test/config/constant.ts，统一从项目根 .env 读取
DEHAZE_HOST / DEHAZE_PASSWORD，避免硬编码。

- BACKENDS：三端后端 base_url（debug 用本机映射端口，与 sdk-js/test 一致）
- REDIS / MYSQL：直连远程基础设施（不依赖本地 docker）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

# 项目根目录（utils/config.py → utils/ → dehaze-test/ → 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 加载 .env 到 os.environ（dehaze-python venv 自带 python-dotenv）
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

DEHAZE_HOST = "127.0.0.1"
DEHAZE_PASSWORD = "Dehaze2026"

# SQL 资源目录（rebuild_mysql 用）
SQL_SCHEMA_DIR = PROJECT_ROOT / "config" / "sql" / "schema"
SQL_DATA_DIR = PROJECT_ROOT / "config" / "sql" / "data"


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str


# 三端后端：本机映射端口（与 dehaze-sdk-js/test/config/constant.ts 一致）
BACKENDS: dict[str, BackendConfig] = {
    "java": BackendConfig("dehaze-java", "http://127.0.0.1:8989"),
    "go": BackendConfig("dehaze-go", "http://127.0.0.1:8990"),
    "python": BackendConfig("dehaze-python", "http://127.0.0.1:8991"),
}

DEFAULT_BACKEND = "java"
DEFAULT_USERNAME = "admin"
# 与项目根 .env 的 DEHAZE_PASSWORD 一致（所有基础设施 + 用户密码统一）
DEFAULT_PASSWORD = os.environ.get("DEHAZE_PASSWORD", "Dehaze2026")

# 三端统一使用 "00000" 作为成功码（对齐 dehaze-sdk-js/src/enums/ResultEnum.ts）
SUCCESS_CODE = "00000"


def get_backend(name: str | None = None) -> BackendConfig:
    """按名称获取后端配置，未指定时返回 DEFAULT_BACKEND。"""
    key = (name or DEFAULT_BACKEND).lower()
    if key not in BACKENDS:
        raise ValueError(f"未知后端: {name}，可选: {list(BACKENDS.keys())}")
    return BACKENDS[key]


# Redis / MySQL 直连配置（与 dehaze-sdk-js/test/utils/{redis,mysql}.ts 对齐）
REDIS_HOST = DEHAZE_HOST
REDIS_PORT = 6379
REDIS_PASSWORD = DEHAZE_PASSWORD
REDIS_DB = 0

MYSQL_HOST = DEHAZE_HOST
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = DEHAZE_PASSWORD
MYSQL_DATABASE = "dehaze"
MYSQL_DATABASE_TEST = "dehaze_test"

# 端口 → 后端名映射（对齐 sdk-js/test/config/constant.ts）
PORT_TO_BACKEND = {
    "8989": "dehaze-java",
    "8990": "dehaze-go",
    "8991": "dehaze-python",
}
