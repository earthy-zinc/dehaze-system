from flasgger import Swagger
from flask import Flask
from flask_pymongo import PyMongo
from flask_sqlalchemy import SQLAlchemy
from minio import Minio
from redis import Redis

def init_mysql(app: Flask):
    """
    Initialize MySQL with SQLAlchemy.
    """
    mysql = SQLAlchemy()
    mysql.init_app(app)
    app.extensions["mysql"] = mysql


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


def init_mongodb(app: Flask):
    """
    Initialize MongoDB with Flask-PyMongo.
    """
    mongodb = PyMongo()
    mongodb.init_app(app)
    app.extensions["mongodb"] = mongodb


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
    app.extensions["minio_client"] = minio_client


def init_swagger(app: Flask):
    """
    Initialize Swagger with Flasgger.
    """
    swagger = Swagger(app)
    swagger.init_app(app)
    app.extensions["swagger"] = swagger
