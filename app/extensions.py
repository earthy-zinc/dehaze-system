import logging

from flasgger import Swagger
from flask import Flask
from flask_jwt_extended import JWTManager
from flask_pymongo import PyMongo
from flask_sqlalchemy import SQLAlchemy
from minio import Minio
from redis import Redis

mysql = SQLAlchemy()
swagger = Swagger()
mongodb = PyMongo()

logger = logging.getLogger(__name__)

def init_jwt(app: Flask):
    jwt = JWTManager(app)
    return jwt

def init_mysql(app: Flask):
    """
    Initialize MySQL with SQLAlchemy.
    """
    global mysql
    mysql.init_app(app)
    logger.info("Mysql 初始化成功")


def init_redis(app: Flask):
    """
    Initialize Redis client.
    """
    redis_client = Redis(
        host=app.config.get("REDIS_HOST"),
        port=app.config.get("REDIS_PORT"),
        password=app.config.get("REDIS_PASSWORD"),
        db=app.config.get("REDIS_DB")
    )
    app.extensions["redis_client"] = redis_client
    logger.info("Redis 初始化成功")

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
        app.config.get("MINIO_ENDPOINT"),
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
    """
    global swagger
    swagger.init_app(app)
