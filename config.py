
class Config:
    pass


class DevelopmentConfig(Config):
    BASE_URL = "http://localhost:8989/api/v1/files"
    PREDICT_IMAGE_PATH = "D:/data/dehaze/predict"
    SQLALCHEMY_DATABASE_URI = "mysql://root:142536aA@10.16.108.141/dehaze?charset=utf8"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False

class TestingConfig(Config):
    pass

class ProductionConfig(Config):
    pass

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}