import hashlib
import logging
from functools import wraps

from flask import jsonify, request
from flask_jwt_extended import verify_jwt_in_request, get_jwt

from app.utils.code import ResultCode
from app.utils.result import warning
from app.utils.logging import logger

try:
    from app.extensions import redis_client
except ImportError:
    redis_client = None
    logger.warning("Redis 未初始化，防重复提交装饰器将不可用")


def anti_repeat(expire_seconds: int = 5):
    """
    防重复提交装饰器

    使用 JWT 的 jti + 请求方法 + URI 生成唯一 key，
    如果短时间内有相同的请求，则拒绝。

    Args:
        expire_seconds: 防重复的时间窗口，默认 5 秒

    Usage:
        @app.route('/api/submit', methods=['POST'])
        @anti_repeat(expire_seconds=5)
        def submit():
            return success()

        # 或者自定义时间窗口
        @app.route('/api/submit', methods=['POST'])
        @anti_repeat(expire_seconds=10)
        def submit():
            return success()

    注意:
        1. 该装饰器需要 JWT 认证
        2. 该装饰器依赖 Redis
        3. 该装饰器对 GET 请求不生效
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 如果 Redis 不可用，直接放行
            if redis_client is None:
                logger.warning("Redis 不可用，防重复提交装饰器已禁用")
                return f(*args, **kwargs)

            # GET 请求不进行防重复提交
            if request.method == 'GET':
                return f(*args, **kwargs)

            try:
                # 获取 JWT token
                verify_jwt_in_request()
                token = get_jwt()
                jti = token.get('jti')  # JWT 的唯一标识

                if not jti:
                    logger.warning("JWT token 中缺少 jti 字段")
                    return f(*args, **kwargs)

                # 生成唯一 key: jti + method + path + query_params
                # 将请求参数排序后拼接，确保参数顺序不影响 key 的生成
                query_params = request.args.to_dict()
                sorted_params = sorted(query_params.items())
                params_str = '&'.join([f"{k}={v}" for k, v in sorted_params])

                key_parts = [
                    'anti_repeat',
                    jti,
                    request.method,
                    request.path,
                    params_str
                ]

                key_str = ':'.join(key_parts)
                # 使用 MD5 生成短 key
                key_hash = hashlib.md5(key_str.encode()).hexdigest()
                redis_key = f"anti_repeat:{key_hash}"

                # 检查 Redis 中是否已存在该 key
                if redis_client.exists(redis_key):
                    logger.info(f"检测到重复提交，key: {redis_key}")
                    return warning(ResultCode.REPEAT_SUBMIT_ERROR)

                # 设置 key，过期时间为 expire_seconds
                redis_client.setex(redis_key, expire_seconds, '1')
                logger.debug(f"设置防重复提交 key: {redis_key}, 过期时间: {expire_seconds}秒")

                return f(*args, **kwargs)

            except Exception as e:
                # 如果出现异常（例如 JWT 认证失败），不影响原函数执行
                logger.error(f"防重复提交装饰器异常: {str(e)}", exc_info=True)
                return f(*args, **kwargs)

        return decorated_function
    return decorator
