class Config:
    pass


class DevelopmentConfig(Config):
    BASE_URL = "http://localhost:8989/api/v1/files"
    SQLALCHEMY_DATABASE_URI = "mysql://root:123456@localhost/dehaze?charset=utf8"
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
