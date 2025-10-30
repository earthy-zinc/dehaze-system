from functools import wraps

from flask import request, jsonify

from app.utils.jwt_util import jwt_required


def request_params_verify(required_params):
    """
    请求参数验证装饰器
    :param required_params: 必需的参数列表
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({
                    'code': 400,
                    'msg': '请求数据不能为空',
                    'data': None
                }), 400

            for param in required_params:
                if param not in data or not data[param]:
                    return jsonify({
                        'code': 400,
                        'msg': f'参数 {param} 不能为空',
                        'data': None
                    }), 400

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# 重新导出jwt_required装饰器
login_required = jwt_required
