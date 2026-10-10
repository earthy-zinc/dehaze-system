"""调试工具库：三端后端、Redis、MySQL 连接配置。

对齐 dehaze-sdk-js/test/config/constant.ts，统一从项目根 .env 读取
按基础设施分区变量（MYSQL_*/REDIS_*/NGINX_STATIC_*），避免硬编码。

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


def _required(name: str) -> str:
    """读取必需的环境变量，缺失即报错。

    .env 缺失时若回落到兜底值（127.0.0.1 / 生产预留库名 dehaze），会静默连到错误的环境，
    故连接与凭据类变量一律要求根 .env 显式提供。
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}：请确认仓库根 .env 已按环境模板落地（.env.dev.example / .env.pre.example）"
        )
    return value


# 种子账号（admin/root/test 等）口令明文，对应 config/sql/data/sys_user.sql 的 bcrypt；
# 非运行时配置：改 .env 不会改变种子口令，只能改 sql 里的哈希
SEED_PASSWORD = "Dehaze2026"

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
# 新用户默认口令（三端后端由根 .env 的 DEFAULT_PASSWORD 注入并用于建号）
DEFAULT_PASSWORD = _required("DEFAULT_PASSWORD")

# 三端统一使用 "00000" 作为成功码（对齐 dehaze-sdk-js/src/enums/ResultEnum.ts）
SUCCESS_CODE = "00000"


def get_backend(name: str | None = None) -> BackendConfig:
    """按名称获取后端配置，未指定时返回 DEFAULT_BACKEND。"""
    key = (name or DEFAULT_BACKEND).lower()
    if key not in BACKENDS:
        raise ValueError(f"未知后端: {name}，可选: {list(BACKENDS.keys())}")
    return BACKENDS[key]


# Redis / MySQL 直连配置（与 dehaze-sdk-js/test/utils/{redis,mysql}.ts 对齐）
REDIS_HOST = _required("REDIS_HOST")
REDIS_PORT = int(_required("REDIS_PORT"))
REDIS_PASSWORD = _required("REDIS_PASSWORD")
REDIS_DATABASE = int(_required("REDIS_DATABASE"))

MYSQL_HOST = _required("MYSQL_HOST")
MYSQL_PORT = int(_required("MYSQL_PORT"))
MYSQL_USER = _required("MYSQL_USERNAME")
MYSQL_PASSWORD = _required("MYSQL_PASSWORD")
MYSQL_DATABASE = _required("MYSQL_DATABASE")
MYSQL_DATABASE_TEST = "dehaze_test"

# 端口 → 后端名映射（对齐 sdk-js/test/config/constant.ts）
PORT_TO_BACKEND = {
    "8989": "dehaze-java",
    "8990": "dehaze-go",
    "8991": "dehaze-python",
}

# MongoDB / MinIO / Elasticsearch：环境重置（reset_env.py）按命名空间清理时使用，
# 变量名与三端后端的配置项保持一致
MONGODB_HOST = _required("MONGODB_HOST")
MONGODB_PORT = int(_required("MONGODB_PORT"))
MONGODB_USERNAME = _required("MONGODB_USERNAME")
MONGODB_PASSWORD = _required("MONGODB_PASSWORD")
MONGODB_DATABASE = _required("MONGODB_DATABASE")

MINIO_HOST = _required("MINIO_HOST")
MINIO_PORT = int(_required("MINIO_PORT"))
MINIO_ACCESS_KEY = _required("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = _required("MINIO_SECRET_KEY")
MINIO_BUCKET = _required("MINIO_BUCKET")

ES_URL = _required("ES_URL")
ES_USERNAME = _required("ES_USERNAME")
ES_PASSWORD = _required("ES_PASSWORD")
ES_INDEX_PREFIX = _required("ES_INDEX_PREFIX")
