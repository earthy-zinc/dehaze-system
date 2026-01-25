"""
幂等性装饰器

支持从 header 或 query param 获取幂等性 token，
检查是否已处理，如果已处理则返回缓存结果。
"""

import hashlib
import json
from functools import wraps
from typing import Optional, Callable

from flask import jsonify, request

from app.utils.code import ResultCode
from app.utils.result import warning
from app.utils.logging import logger

try:
    from app.extensions import redis_client
except ImportError:
    redis_client = None
    logger.warning("Redis 未初始化，幂等性装饰器将不可用")


def idempotent(
    header_name: str = 'X-Idempotency-Key',
    param_name: str = 'idempotencyKey',
    expire_seconds: int = 86400,
    error_on_duplicate: bool = False,
    result_serializer: Optional[Callable] = None
):
    """
    幂等性装饰器

    支持从 header 或 query param 获取幂等性 token，
    检查是否已处理，如果已处理则返回缓存结果。

    Args:
        header_name: Header 中幂等性 token 的字段名，默认 'X-Idempotency-Key'
        param_name: Query param 中幂等性 token 的参数名，默认 'idempotencyKey'
        expire_seconds: 缓存结果的过期时间（秒），默认 86400（24小时）
        error_on_duplicate: 重复请求时是否返回错误（True）或缓存结果（False），默认 False
        result_serializer: 结果序列化函数，用于将响应序列化为可缓存的格式

    Usage:
        # 基本使用（从 header 获取 token）
        @app.route('/api/create-order', methods=['POST'])
        @idempotent()
        def create_order():
            # 创建订单逻辑
            return success(data={"orderId": "12345"})

        # 请求示例：
        # POST /api/create-order
        # Headers: X-Idempotency-Key: unique-token-123
        # Body: {...}

        # 自定义参数名
        @app.route('/api/payment', methods=['POST'])
        @idempotent(
            header_name='X-Request-ID',
            param_name='requestId',
            expire_seconds=3600
        )
        def payment():
            # 支付逻辑
            return success(data={"paymentId": "67890"})

        # 重复请求时返回错误
        @app.route('/api/strict', methods=['POST'])
        @idempotent(error_on_duplicate=True)
        def strict_operation():
            # 严格操作
            return success(data={"status": "success"})

    注意:
        1. 该装饰器依赖于 Redis
        2. 幂等性 token 的优先级: header > query param > body
        3. 默认情况下，重复请求会返回第一次的响应结果（缓存）
        4. 如果 Redis 不可用，装饰器会跳过幂等性检查（降级处理）
        5. 只对 POST、PUT、PATCH 方法生效，GET、DELETE 等方法不缓存
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 如果 Redis 不可用，跳过幂等性检查（降级处理）
            if redis_client is None:
                logger.warning("Redis 不可用，幂等性装饰器已禁用（降级跳过）")
                return f(*args, **kwargs)

            # 只对需要修改的方法进行幂等性控制
            if request.method not in ['POST', 'PUT', 'PATCH']:
                return f(*args, **kwargs)

            try:
                # 获取幂等性 token
                idempotency_key = _get_idempotency_key(header_name, param_name)

                # 如果没有提供 token，不进行幂等性检查
                if not idempotency_key:
                    logger.info("未提供幂等性 token，跳过幂等性检查")
                    return f(*args, **kwargs)

                # 生成唯一的缓存 key
                cache_key = _generate_cache_key(request.path, request.method, idempotency_key)

                # 检查是否已处理
                cached_result = redis_client.get(cache_key)

                if cached_result:
                    logger.info(f"检测到重复请求，返回缓存结果，key: {cache_key}")

                    if error_on_duplicate:
                        # 返回错误
                        return warning(ResultCode.REPEAT_SUBMIT_ERROR)
                    else:
                        # 返回缓存结果
                        try:
                            cached_data = json.loads(cached_result)
                            return jsonify(cached_data)
                        except json.JSONDecodeError:
                            logger.error("缓存结果反序列化失败，执行原始请求")
                            return f(*args, **kwargs)

                # 执行原始函数
                response = f(*args, **kwargs)

                # 只缓存成功的响应（HTTP 状态码为 2xx）
                if _is_success_response(response):
                    # 序列化响应数据
                    response_data = _serialize_response(response, result_serializer)

                    # 缓存响应结果
                    redis_client.setex(cache_key, expire_seconds, json.dumps(response_data))
                    logger.debug(f"缓存响应结果，key: {cache_key}, 过期时间: {expire_seconds}秒")

                return response

            except Exception as e:
                # 如果出现异常，记录日志但不影响请求
                logger.error(f"幂等性装饰器异常: {str(e)}", exc_info=True)
                # 继续执行原始请求
                return f(*args, **kwargs)

        return decorated_function
    return decorator


def _get_idempotency_key(header_name: str, param_name: str) -> Optional[str]:
    """
    获取幂等性 token

    优先级: header > query param > body

    Args:
        header_name: Header 字段名
        param_name: Query param 字段名

    Returns:
        幂等性 token 字符串
    """
    # 1. 从 header 获取
    idempotency_key = request.headers.get(header_name)
    if idempotency_key:
        return str(idempotency_key).strip()

    # 2. 从 query param 获取
    idempotency_key = request.args.get(param_name)
    if idempotency_key:
        return str(idempotency_key).strip()

    # 3. 从 body 获取（仅 JSON 请求）
    if request.is_json:
        data = request.get_json()
        if data:
            idempotency_key = data.get(param_name)
            if idempotency_key:
                return str(idempotency_key).strip()

    # 4. 从 form data 获取
    idempotency_key = request.form.get(param_name)
    if idempotency_key:
        return str(idempotency_key).strip()

    return None


def _generate_cache_key(path: str, method: str, idempotency_key: str) -> str:
    """
    生成缓存 key

    Args:
        path: 请求路径
        method: 请求方法
        idempotency_key: 幂等性 token

    Returns:
        Redis 缓存 key
    """
    # 生成短 key，避免 key 过长
    key_string = f"{method}:{path}:{idempotency_key}"
    key_hash = hashlib.md5(key_string.encode()).hexdigest()

    return f"idempotent:{key_hash}"


def _is_success_response(response) -> bool:
    """
    判断响应是否成功

    Args:
        response: Flask 响应对象

    Returns:
        是否成功（HTTP 状态码为 2xx）
    """
    try:
        # 如果是 Response 对象
        if hasattr(response, 'status_code'):
            return 200 <= response.status_code < 300

        # 如果是元组 (response, status_code)
        if isinstance(response, tuple) and len(response) >= 2:
            status_code = response[1]
            return 200 <= status_code < 300

        # 默认认为成功
        return True

    except Exception:
        return True


def _serialize_response(response, serializer: Optional[Callable] = None) -> dict:
    """
    序列化响应数据

    Args:
        response: Flask 响应对象
        serializer: 自定义序列化函数

    Returns:
        序列化后的字典
    """
    try:
        # 如果提供了自定义序列化函数
        if serializer:
            return serializer(response)

        # 解析响应
        if hasattr(response, 'get_json'):
            # Response 对象
            data = response.get_json()
            if data:
                return data

        elif isinstance(response, tuple) and len(response) >= 1:
            # 元组 (response, status_code, headers)
            first_element = response[0]

            if hasattr(first_element, 'get_json'):
                data = first_element.get_json()
                if data:
                    return data
            elif isinstance(first_element, (dict, list, str)):
                return {"data": first_element}

        # 默认返回空字典
        return {"data": response}

    except Exception as e:
        logger.error(f"序列化响应数据失败: {str(e)}", exc_info=True)
        return {"data": None}
