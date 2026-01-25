import logging

from flasgger import Swagger
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_pymongo import PyMongo
from flask_sqlalchemy import SQLAlchemy
from minio import Minio
from redis import Redis

mysql = SQLAlchemy()
migrate = Migrate()
swagger = Swagger()
mongodb = PyMongo()
redis_client = None  # 全局Redis客户端
limiter = None  # 全局限流器

logger = logging.getLogger(__name__)


def init_jwt(app: Flask):
    jwt = JWTManager(app)
    return jwt


def init_mysql(app: Flask):
    """
    Initialize MySQL with SQLAlchemy.
    """
    global mysql, migrate
    mysql.init_app(app)
    migrate.init_app(app, mysql)
    logger.info("Mysql 初始化成功")


def init_redis(app: Flask):
    """
    Initialize Redis client.
    """
    global redis_client
    redis_client = Redis(
        host=app.config.get("REDIS_HOST"),
        port=app.config.get("REDIS_PORT"),
        password=app.config.get("REDIS_PASSWORD"),
        db=app.config.get("REDIS_DB")
    )
    app.extensions["redis_client"] = redis_client
    logger.info("Redis 初始化成功")


def init_limiter(app: Flask):
    """
    Initialize Flask-Limiter for rate limiting.

    使用 Redis 作为后端存储，支持滑动窗口算法。
    配置说明：
        - 默认使用客户端 IP 作为限流 key
        - 使用 Redis 存储限流计数
        - 支持滑动窗口、固定窗口、令牌桶等多种算法
    """
    global limiter

    # 测试环境禁用限流
    if app.config.get("TESTING"):
        logger.info("测试环境：Flask-Limiter 已禁用")
        app.extensions["limiter"] = None
        return

    # 构建 Redis URI
    redis_host = app.config.get("REDIS_HOST", "localhost")
    redis_port = app.config.get("REDIS_PORT", 6379)
    redis_password = app.config.get("REDIS_PASSWORD", "")
    redis_db = app.config.get("REDIS_DB", 0)

    if redis_password:
        storage_uri = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
    else:
        storage_uri = f"redis://{redis_host}:{redis_port}/{redis_db}"

    limiter = Limiter(
        key_func=get_remote_address,  # 默认使用 IP 限流
        storage_uri=storage_uri,
        strategy="fixed-window",  # 固定窗口策略（新版本移除了 elastic-expiry）
        default_limits=["200 per day", "50 per hour"]  # 默认全局限制（可选）
    )
    limiter.init_app(app)
    app.extensions["limiter"] = limiter
    logger.info("Flask-Limiter 初始化成功")


def init_mongodb(app: Flask):
    """
    Initialize MongoDB with Flask-PyMongo.
    """
    global mongodb
    mongodb.init_app(app)
    logger.info("MongoDB 初始化成功")


def init_minio(app: Flask):
    """
    Initialize MinIO client.
    """
    minio_client = Minio(
        endpoint=app.config.get("MINIO_ENDPOINT"),
        access_key=app.config.get("MINIO_ACCESS_KEY"),
        secret_key=app.config.get("MINIO_SECRET_KEY"),
        secure=app.config.get("MINIO_SECURE")
    )
    # Ensure bucket exists
    bucket_name = app.config.get("MINIO_BUCKET_NAME")
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        minio_client.set_bucket_policy(bucket_name, policy="public-read")

    app.extensions["minio_client"] = minio_client
    logger.info("MinIO 初始化成功")


def init_swagger(app: Flask):
    """
    Initialize Swagger with Flasgger.
    
    注意: 使用 flask-openapi3 的 APIBlueprint 会自动注册到 /openapi 路径，
    而 flasgger 仍然使用 /apidocs 路径。两者可以共存。
    """
    global swagger
    swagger.init_app(app)
