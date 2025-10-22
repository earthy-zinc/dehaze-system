from flask import Flask
from flask_socketio import SocketIO

from app.utils.error_handlers import register_error_handlers
from app.utils.logging import setup_logging
from config import config


def create_app(config_name: str):
    setup_logging()
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    # 初始化扩展
    init_extensions(app)
    # 初始化SocketIO
    socketio = SocketIO(cors_allowed_origins="*")
    socketio.init_app(app)
    # 初始化WebSocket服务
    from app.service.websocket_service import WebSocketService
    WebSocketService(socketio)
    # 注册错误处理
    register_error_handlers(app)
    # 注册路由
    from app.route import init_routes
    init_routes(app)
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
    # init_minio(app)
    init_swagger(app)
    init_jwt(app)
