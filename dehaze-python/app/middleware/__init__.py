"""
中间件层

仅包含中间件逻辑，不包含数据模型或服务层代码
"""

from app.middleware.operation_log import init_operation_log

__all__ = [
    'init_operation_log',
]
