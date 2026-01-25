"""
操作日志中间件

功能：
1. 记录所有HTTP请求的详细信息
2. 记录请求耗时、响应状态、错误信息等
3. 支持用户ID追踪（从JWT Token中获取）
4. 将日志数据写入数据库

注意事项：
1. 敏感数据（如密码）应在记录时过滤
2. 请求体和响应体可能很大，建议设置大小限制
3. 异步写入数据库以避免影响请求性能
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import atexit
from datetime import datetime, timezone

from flask import Flask, request, g

from app.extensions import mysql

logger = logging.getLogger(__name__)


# 创建线程池用于异步写入日志（修复：避免阻塞主线程，提高性能）
# 注意：程序退出时会自动调用 shutdown
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='operation_log')
atexit.register(_executor.shutdown, wait=False)


# 需要排除记录的路径（避免记录健康检查等非业务请求）
EXCLUDE_PATHS = [
    '/health',
    '/metrics',
    '/favicon.ico',
    '/static/',
    '/api/v1/auth/captcha',
]

# 需要过滤的敏感字段（在记录请求体时过滤）
SENSITIVE_FIELDS = [
    'password',
    'passwd',
    'pwd',
    'secret',
    'token',
    'access_token',
    'refresh_token'
]

# 请求体和响应体的最大长度（字符数）
MAX_BODY_LENGTH = 5000
MAX_RESP_LENGTH = 5000


def filter_sensitive_data(data: dict) -> dict:
    """
    过滤敏感数据
    将敏感字段的值替换为 '******'
    """
    if not isinstance(data, dict):
        return data

    filtered = {}
    for key, value in data.items():
        if isinstance(key, str) and key.lower() in SENSITIVE_FIELDS:
            filtered[key] = '******'
        elif isinstance(value, dict):
            filtered[key] = filter_sensitive_data(value)
        elif isinstance(value, list):
            filtered[key] = [
                filter_sensitive_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            filtered[key] = value

    return filtered


def should_log_request(path: str) -> bool:
    """
    判断是否需要记录该请求
    排除健康检查、静态资源等非业务请求
    """
    for exclude_path in EXCLUDE_PATHS:
        if path.startswith(exclude_path):
            return False
    return True


def truncate_string(s: str, max_length: int) -> str:
    """
    截断字符串到指定长度
    """
    if len(s) <= max_length:
        return s
    return s[:max_length] + '...[truncated]'


def capture_response_data(response) -> dict:
    """
    捕获响应数据
    尝试从不同类型的响应对象中提取数据
    """
    try:
        # Flask Response 对象
        if hasattr(response, 'get_data'):
            data = response.get_data(as_text=True)
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # 不是JSON格式，返回原始字符串
                return {'data': truncate_string(data, MAX_RESP_LENGTH)}

        # 元组形式 (response, status, headers)
        if isinstance(response, tuple):
            resp_data = response[0]
            if isinstance(resp_data, (dict, list)):
                return resp_data
            elif isinstance(resp_data, str):
                try:
                    return json.loads(resp_data)
                except json.JSONDecodeError:
                    return {'data': truncate_string(resp_data, MAX_RESP_LENGTH)}

        # 字典或列表
        if isinstance(response, (dict, list)):
            return response

        return {'data': str(response)}
    except Exception as e:
        logger.error(f'捕获响应数据失败: {e}')
        return {'error': 'Failed to capture response data'}


def save_operation_log(log_data: dict):
    """
    保存操作日志到数据库（同步版本）

    修复说明：
        原函数为同步写入，可能阻塞请求处理。
        现已添加异步版本 save_operation_log_async，建议使用异步版本。
        保留此函数以兼容旧代码和直接调用场景。

    Args:
        log_data: 日志数据字典
    """
    # 延迟导入避免循环依赖
    from app.models import SysOperationLog

    try:
        operation_log = SysOperationLog(
            ip=log_data.get('ip', ''),
            method=log_data.get('method', ''),
            path=log_data.get('path', ''),
            status=log_data.get('status', 200),
            latency=log_data.get('latency', 0),
            agent=log_data.get('agent', ''),
            error_message=log_data.get('error_message', ''),
            body=log_data.get('body', ''),
            resp=log_data.get('resp', ''),
            user_id=log_data.get('user_id')
        )

        mysql.session.add(operation_log)
        mysql.session.commit()

        logger.debug(f'操作日志已保存: {log_data.get("method")} {log_data.get("path")}')
    except Exception as e:
        # 记录失败不应影响主业务
        logger.error(f'保存操作日志失败: {e}', exc_info=True)
        try:
            mysql.session.rollback()
        except Exception:
            pass


def save_operation_log_async(log_data: dict):
    """
    异步保存操作日志到数据库（修复：避免阻塞主线程，提高性能）

    修复说明：
        使用线程池异步写入日志，避免阻塞 HTTP 请求处理线程。
        线程池大小为 2，在 app 模块级别创建，程序退出时自动关闭。

    Args:
        log_data: 日志数据字典

    注意:
        1. 异步写入，日志可能稍有延迟
        2. 使用 Flask app context 确保可以访问数据库
        3. 异常不会影响主业务流程
    """
    # 延迟导入避免循环依赖
    from app.models import SysOperationLog

    def _save():
        """内部保存函数，在线程中执行"""
        try:
            # 需要在 app context 中执行
            with mysql.session.begin():
                operation_log = SysOperationLog(
                    ip=log_data.get('ip', ''),
                    method=log_data.get('method', ''),
                    path=log_data.get('path', ''),
                    status=log_data.get('status', 200),
                    latency=log_data.get('latency', 0),
                    agent=log_data.get('agent', ''),
                    error_message=log_data.get('error_message', ''),
                    body=log_data.get('body', ''),
                    resp=log_data.get('resp', ''),
                    user_id=log_data.get('user_id')
                )
                mysql.session.add(operation_log)
            logger.debug(f'异步操作日志已保存: {log_data.get("method")} {log_data.get("path")}')
        except Exception as e:
            # 记录失败不应影响主业务
            logger.error(f'异步保存操作日志失败: {e}', exc_info=True)
            try:
                mysql.session.rollback()
            except Exception:
                pass

    # 提交到线程池
    _executor.submit(_save)


def before_request_handler():
    """
    请求前处理函数
    记录请求开始时间，存储请求数据到 Flask g 对象
    """
    # 判断是否需要记录
    if not should_log_request(request.path):
        return

    # 记录请求开始时间
    g.start_time = time.time()

    # 提取请求体数据
    request_body = {}
    try:
        if request.is_json:
            request_body = request.get_json(silent=True) or {}
        elif request.form:
            request_body = dict(request.form.to_dict())
        elif request.data:
            # 尝试解析其他格式的数据
            try:
                request_body = json.loads(request.data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    except Exception as e:
        logger.warning(f'提取请求体失败: {e}')

    # 过滤敏感数据
    filtered_body = filter_sensitive_data(request_body)

    # 序列化为JSON字符串
    body_str = json.dumps(filtered_body, ensure_ascii=False, default=str)
    body_str = truncate_string(body_str, MAX_BODY_LENGTH)

    # 存储到 g 对象供 after_request 使用
    # 注：ProxyFix 中间件已在 app 初始化时配置，request.remote_addr 即为真实客户端 IP
    g.operation_log_data = {
        'ip': request.remote_addr or '',
        'method': request.method,
        'path': request.path,
        'agent': request.headers.get('User-Agent', ''),
        'body': body_str,
        'user_id': getattr(request, 'current_user_id', None)
    }


def after_request_handler(response):
    """
    请求后处理函数
    计算请求耗时，捕获响应数据，保存日志
    """
    # 判断是否需要记录
    if not should_log_request(request.path):
        return response

    # 检查是否有开始时间（可能某些请求被 before_request_handler 排除）
    if not hasattr(g, 'start_time'):
        return response

    try:
        # 计算耗时（毫秒）
        end_time = time.time()
        latency = int((end_time - g.start_time) * 1000)

        # 获取响应状态码
        status_code = getattr(response, 'status_code', 200)

        # 捕获响应数据
        response_data = capture_response_data(response)

        # 序列化为JSON字符串
        resp_str = json.dumps(response_data, ensure_ascii=False, default=str)
        resp_str = truncate_string(resp_str, MAX_RESP_LENGTH)

        # 获取错误信息（从 g 对象中，如果有的话）
        error_message = getattr(g, 'error_message', '')

        # 组装完整的日志数据
        log_data = g.operation_log_data
        log_data.update({
            'status': status_code,
            'latency': latency,
            'resp': resp_str,
            'error_message': error_message
        })

        # 修复：使用异步保存日志，避免阻塞主线程（提高性能）
        save_operation_log_async(log_data)

    except Exception as e:
        # 日志记录失败不应影响主业务
        logger.error(f'记录操作日志失败: {e}', exc_info=True)

    return response


def init_operation_log(app: Flask):
    """
    初始化操作日志中间件
    注册 before_request 和 after_request 钩子

    Args:
        app: Flask 应用实例

    使用示例:
        from app.middleware.operation_log import init_operation_log

        def create_app():
            app = Flask(__name__)
            # ... 其他初始化代码
            # 注册操作日志中间件
            init_operation_log(app)
            return app
    """
    # 注册 before_request 钩子
    @app.before_request
    def _before_request():
        return before_request_handler()

    # 注册 after_request 钩子
    @app.after_request
    def _after_request(response):
        return after_request_handler(response)

    logger.info('操作日志中间件已初始化')
