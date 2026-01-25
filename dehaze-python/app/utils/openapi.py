"""
OpenAPI 工具模块 - 提供 flask-openapi3 相关的工具函数和配置
"""
from functools import wraps
from typing import Callable, Any

from flask import g
from flask_openapi3 import Info, OpenAPI, APIBlueprint, SecurityScheme
from pydantic import ValidationError

from app.utils.result import error


# OpenAPI 配置
info = Info(
    title="Dehaze API",
    version="1.0.0",
    description="图像去雾系统 API 文档"
)

# JWT Bearer 安全方案
jwt_security = SecurityScheme(type="http", scheme="bearer", bearerFormat="JWT")
security_schemes = {"BearerAuth": jwt_security}


def create_openapi_app(import_name: str, **kwargs) -> OpenAPI:
    """
    创建带有 OpenAPI 配置的 Flask 应用
    
    Args:
        import_name: 应用导入名称
        **kwargs: 其他 Flask 配置参数
        
    Returns:
        OpenAPI: 配置好的 OpenAPI 应用实例
    """
    app = OpenAPI(
        import_name,
        info=info,
        security_schemes=security_schemes,
        **kwargs
    )
    return app


def create_api_blueprint(name: str, import_name: str, url_prefix: str = None, **kwargs) -> APIBlueprint:
    """
    创建带有安全配置的 API Blueprint
    
    Args:
        name: Blueprint 名称
        import_name: 导入名称
        url_prefix: URL 前缀
        **kwargs: 其他配置参数
        
    Returns:
        APIBlueprint: 配置好的 API Blueprint
    """
    return APIBlueprint(
        name,
        import_name,
        url_prefix=url_prefix,
        abp_security=[{"BearerAuth": []}],
        **kwargs
    )


def validate_request(func: Callable) -> Callable:
    """
    请求验证装饰器 - 处理 Pydantic 验证错误
    
    用法:
        @validate_request
        def my_endpoint(query: MyQuery):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                errors.append(f"{field}: {err['msg']}")
            return error("; ".join(errors), 400)
    return wrapper
