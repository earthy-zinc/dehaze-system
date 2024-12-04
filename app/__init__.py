from flask import Flask
from config import config
from app.route.model import model_blueprint
from app.utils.error_handlers import register_error_handlers


def create_app(config_name: str):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    # 初始化扩展
    init_extensions(app)
    # 注册错误处理
    register_error_handlers(app)
    # 注册蓝图
    app.register_blueprint(model_blueprint)
    return app


def init_extensions(app: Flask):
    """
    Initialize all extensions in a centralized manner.
    """
    from app.extensions import init_mysql, init_redis, init_mongodb, init_minio, init_swagger, init_jwt

    # 初始化每个依赖
    init_mysql(app)
    init_redis(app)
    init_mongodb(app)
    init_minio(app)
    init_swagger(app)
    init_jwt(app)
