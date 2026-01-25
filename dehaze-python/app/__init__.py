from flask import Flask
from flask_openapi3 import OpenAPI, Info, SecurityScheme
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix

from app.utils.error_handlers import register_error_handlers
from app.utils.logging import setup_logging
from config import config

# OpenAPI 配置
openapi_info = Info(
    title="Dehaze API",
    version="1.0.0",
    description="图像去雾系统 API 文档"
)

# JWT Bearer 安全方案
jwt_security = SecurityScheme(type="http", scheme="bearer", bearerFormat="JWT")
security_schemes = {"BearerAuth": jwt_security}


def create_app(config_name: str):
    setup_logging()
    # 使用 OpenAPI 替代 Flask，以支持自动生成 API 文档
    app = OpenAPI(
        __name__,
        info=openapi_info,
        security_schemes=security_schemes,
        doc_prefix="/openapi"  # OpenAPI 文档路径前缀，与 flasgger 的 /apidocs 区分
    )
    app.config.from_object(config[config_name])
    # 配置代理信任：支持 X-Forwarded-For/Proto/Host/Port/Prefix
    # x_for=1 表示信任一层代理的 X-Forwarded-For，之后 request.remote_addr 即为真实客户端 IP
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
    # 禁用严格的斜杠检查，避免 /api/v1/datasets 和 /api/v1/datasets/ 的重定向问题
    app.url_map.strict_slashes = False
    # 初始化扩展
    init_extensions(app)
    # 设置任务执行器的app上下文
    from app.service.task_service import ThreadedTaskExecutor
    ThreadedTaskExecutor.set_app_context(app)
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
    from app.extensions import init_mysql, init_redis, init_mongodb, init_minio, init_swagger, init_jwt, init_limiter

    # 初始化每个依赖
    init_mysql(app)
    init_redis(app)
    init_limiter(app)  # 限流器初始化（依赖 Redis 配置，需在 init_redis 之后）
    init_mongodb(app)
    # init_minio(app)
    init_swagger(app)
    init_jwt(app)
