from functools import wraps

import jwt
from flask import request, current_app

from app.utils.result import error


def jwt_required(f):
    """
    JWT认证装饰器
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None

        # 检查请求头中的Authorization字段
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                # 提取Bearer token
                token = auth_header.split(" ")[1]
            except IndexError:
                return error('无效的认证令牌格式', 401)

        # 如果没有token，返回错误
        if not token:
            return error('缺少认证令牌', 401)

        try:
            # 解码token
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            # 将用户ID存储在请求上下文中
            request.current_user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return error('令牌已过期', 401)
        except jwt.InvalidTokenError:
            return error('无效的令牌', 401)

        return f(*args, **kwargs)

    return decorated_function


def get_current_user_id():
    """
    获取当前用户ID
    """
    return getattr(request, 'current_user_id', None)
