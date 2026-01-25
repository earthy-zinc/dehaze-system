from functools import wraps

from flask import request

from app.utils.code import ResultCode
from app.utils.result import warning
from app.utils.logging import logger

try:
    from app.extensions import redis_client
except ImportError:
    redis_client = None
    logger.warning("Redis 未初始化，验证码校验装饰器将不可用")


def verify_captcha(
    captcha_key_param: str = 'captchaKey',
    captcha_code_param: str = 'captchaCode',
    redis_key_prefix: str = 'captcha',
    error_on_missing: bool = True
):
    """
    验证码校验装饰器

    从请求参数中获取验证码 key 和验证码，从 Redis 中比对。
    验证成功后删除已使用的验证码。

    Args:
        captcha_key_param: 验证码 key 的参数名，默认 'captchaKey'
        captcha_code_param: 验证码的参数名，默认 'captchaCode'
        redis_key_prefix: Redis 中验证码的 key 前缀，默认 'captcha'
        error_on_missing: 当缺少验证码参数时是否报错，默认 True

    Usage:
        # 基本使用
        @app.route('/api/login', methods=['POST'])
        @verify_captcha()
        def login():
            return success()

        # 自定义参数名
        @app.route('/api/login', methods=['POST'])
        @verify_captcha(
            captcha_key_param='verify_key',
            captcha_code_param='verify_code'
        )
        def login():
            return success()

    注意:
        1. 该装饰器依赖于 Redis
        2. 验证成功后，验证码会被删除（一次性使用）
        3. 如果 Redis 不可用，装饰器会跳过验证（降级处理）
        4. 验证码比对不区分大小写
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 如果 Redis 不可用，跳过验证（降级处理）
            if redis_client is None:
                logger.warning("Redis 不可用，验证码校验装饰器已禁用（降级跳过）")
                return f(*args, **kwargs)

            try:
                # 获取请求参数
                if request.method == 'GET':
                    captcha_key = request.args.get(captcha_key_param)
                    captcha_code = request.args.get(captcha_code_param)
                else:  # POST
                    # 支持 JSON 和 Form Data
                    if request.is_json:
                        data = request.get_json()
                        captcha_key = data.get(captcha_key_param) if data else None
                        captcha_code = data.get(captcha_code_param) if data else None
                    else:
                        captcha_key = request.form.get(captcha_key_param)
                        captcha_code = request.form.get(captcha_code_param)

                # 检查参数是否存在
                if not captcha_key or not captcha_code:
                    if error_on_missing:
                        logger.warning("验证码参数缺失")
                        return warning(ResultCode.PARAM_IS_NULL)
                    else:
                        logger.info("验证码参数缺失，跳过验证")
                        return f(*args, **kwargs)

                # 构建 Redis key
                redis_key = f"{redis_key_prefix}:{captcha_key}"

                # 从 Redis 获取验证码
                stored_code = redis_client.get(redis_key)

                if not stored_code:
                    logger.warning(f"验证码不存在或已过期，key: {captcha_key}")
                    return warning(ResultCode.VERIFY_CODE_TIMEOUT)

                # 转换为字符串并去除空白
                stored_code_str = stored_code.decode('utf-8').strip() if isinstance(stored_code, bytes) else str(stored_code).strip()
                captcha_code_str = str(captcha_code).strip()

                # 验证码比对（不区分大小写）
                if stored_code_str.lower() != captcha_code_str.lower():
                    logger.warning(
                        f"验证码错误，期望: {stored_code_str}, "
                        f"实际: {captcha_code_str}, key: {captcha_key}"
                    )
                    return warning(ResultCode.VERIFY_CODE_ERROR)

                # 验证成功，删除已使用的验证码
                redis_client.delete(redis_key)
                logger.debug(f"验证码验证成功，已删除验证码，key: {captcha_key}")

                return f(*args, **kwargs)

            except Exception as e:
                # 如果出现异常，记录日志但不影响请求
                logger.error(f"验证码校验装饰器异常: {str(e)}", exc_info=True)
                # 可以选择继续执行或返回错误，这里选择继续执行
                return f(*args, **kwargs)

        return decorated_function
    return decorator
