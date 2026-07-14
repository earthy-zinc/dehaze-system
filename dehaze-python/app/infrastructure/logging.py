import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional

from pythonjsonlogger.json import JsonFormatter as BaseJsonFormatter

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


class UTF8RotatingFileHandler(RotatingFileHandler):
    """支持 UTF-8 编码的轮转文件处理器"""

    def __init__(self, filename, mode='a', max_bytes=0, backup_count=0, encoding='utf-8', delay=False):
        # 确保日志目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(
            filename) else '.', exist_ok=True)
        super().__init__(filename, mode, max_bytes, backup_count, encoding, delay)


class UTF8TimedRotatingFileHandler(TimedRotatingFileHandler):
    """支持 UTF-8 编码的定时轮转文件处理器"""

    def __init__(self, filename, when='midnight', interval=1, backup_count=0,
                 encoding='utf-8', delay=False, utc=False, at_time=None):
        # 确保日志目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(
            filename) else '.', exist_ok=True)
        super().__init__(filename, when, interval,
                         backup_count, encoding, delay, utc, at_time)


class JsonFormatter(BaseJsonFormatter):
    """JSON 结构化日志格式化器（支持 TraceID 注入）"""

    def add_fields(self, log_data, record, message_dict):
        super().add_fields(log_data, record, message_dict)
        from app.config import settings
        log_data["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["service"] = settings.APP_NAME
        log_data["thread"] = record.thread
        log_data["trace_id"] = _trace_id_var.get("")


class TraceIDFilter(logging.Filter):
    """TraceID 注入过滤器（用于文本格式日志）"""

    def filter(self, record):
        record.trace_id = _trace_id_var.get("")
        return True


def setup_logging(use_json_format: Optional[bool] = None):
    """
    设置日志记录系统

    所有配置从 config.settings 读取，仅 use_json_format 可外部覆盖（用于测试）。
    """
    from app.config import settings

    log_level = settings.LOG_LEVEL
    log_format = settings.LOG_FORMAT
    date_format = settings.LOG_DATE_FORMAT
    log_file = settings.LOG_FILE
    log_dir = settings.LOG_DIR
    max_bytes = settings.LOG_MAX_BYTES
    backup_count = settings.LOG_BACKUP_COUNT
    enable_console = settings.LOG_ENABLE_CONSOLE
    enable_file = settings.LOG_ENABLE_FILE
    rotation_type = settings.LOG_ROTATION_TYPE
    use_json_format = use_json_format if use_json_format is not None else settings.LOG_FORMAT_JSON

    # 转换日志级别字符串为 logging 常量
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)

    # 清除现有的handlers避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # 创建格式化器
    # TraceIDFilter 必须挂载到 handler 而非 logger：
    # Python logging 的 filter 只对直接挂载的 logger 生效，子 logger 的日志在传播到 root handler 时
    # record 上没有 trace_id 属性，会导致 %(trace_id)s 格式化失败（KeyError）
    trace_filter = TraceIDFilter()

    # 文件输出始终使用 JSON 结构化格式（供日志集中化管道 ELK/Loki 采集）
    file_formatter: logging.Formatter = JsonFormatter()
    # 控制台输出：JSON 模式下使用 JSON，否则使用人类可读文本
    if use_json_format:
        console_formatter: logging.Formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(log_format, date_format)

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(trace_filter)
        root_logger.addHandler(console_handler)

    if enable_file:
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 默认日志文件路径
        if log_file is None:
            log_file = os.path.join(log_dir, "app.log")
        elif not os.path.isabs(log_file):
            log_file = os.path.join(log_dir, log_file)

        # 根据轮转类型选择处理器
        if rotation_type == "time":
            # 基于时间的轮转 (每天午夜轮转)
            file_handler: logging.Handler = UTF8TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                backup_count=backup_count
            )
        else:
            # 基于大小的轮转
            file_handler = UTF8RotatingFileHandler(
                log_file,
                max_bytes=max_bytes,
                backup_count=backup_count
            )

        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(trace_filter)
        root_logger.addHandler(file_handler)

    root_logger.info("日志系统初始化成功")
    root_logger.info(f"日志级别: {log_level}")
    root_logger.info(f"控制台格式: {'JSON' if use_json_format else '文本'}")
    root_logger.info("文件格式: JSON")
    if enable_file:
        root_logger.info(f"日志文件: {log_file}")
        root_logger.info(f"轮转类型: {rotation_type}")
    if enable_console:
        root_logger.info("控制台输出: 启用")


logger = logging.getLogger(__name__)
