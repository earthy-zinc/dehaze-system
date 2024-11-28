import torch
import os.path as path


class Config:
    DEVICE_ID = [0]
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 获取当前项目目录
    PROJECT_PATH = path.dirname(path.abspath(__file__))
    DATASET_PATH = "D:/DeepLearning/dataset/"
    MODEL_PATH = path.join(PROJECT_PATH, "trained_model")


class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'
    BASE_URL = "http://localhost:8989/api/v1/files"

    DATASET_PATH = "D:/DeepLearning/dataset"
    DATASET_THUMBNAIL_PATH = "D:/DeepLearning/thumbnail"

    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@localhost/dehaze?charset=utf8"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_PASSWORD = '123456'
    REDIS_DB = 0

    MONGO_URI = "mongodb://localhost:27017/"

    MINIO_ENDPOINT = "localhost:9000"
    MINIO_ACCESS_KEY = "admin"
    MINIO_SECRET_KEY = "12345678"
    MINIO_SECURE = False
    MINIO_BUCKET_NAME = "trained-models"


class TestingConfig(Config):
    TESTING = True


class ProductionConfig(Config):
    SECRET_KEY = "1234"
    BASE_URL = "http://dehaze-python/api/v1/files"

    DATASET_PATH = "/app/dataset"
    DATASET_THUMBNAIL_PATH = "/app/thumbnail"

    SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root:123456@192.168.31.3/dehaze?charset=utf8"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

    REDIS_HOST = '192.168.31.3'
    REDIS_PORT = 6379
    REDIS_PASSWORD = '123456'
    REDIS_DB = 0

    MONGO_URI = "mongodb://192.168.31.3:27017/"

    MINIO_ENDPOINT = "192.168.31.3:9000"
    MINIO_ACCESS_KEY = "admin"
    MINIO_SECRET_KEY = "12345678"
    MINIO_SECURE = False
    MINIO_BUCKET_NAME = "trained-models"


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
