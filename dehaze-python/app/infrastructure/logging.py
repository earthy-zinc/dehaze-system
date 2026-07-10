import logging
import os
import sys
from contextvars import ContextVar
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
        log_data["timestamp"] = record.created
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["service"] = "dehaze-python"
        log_data["thread"] = record.thread
        log_data["trace_id"] = _trace_id_var.get("")


class TraceIDFilter(logging.Filter):
    """TraceID 注入过滤器（用于文本格式日志）"""

    def filter(self, record):
        record.trace_id = _trace_id_var.get("")
        return True


def setup_logging(
        log_level: Optional[str] = None,
        log_format: Optional[str] = None,
        date_format: Optional[str] = None,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
        enable_console: Optional[bool] = None,
        enable_file: Optional[bool] = None,
        rotation_type: Optional[str] = None,
        use_json_format: Optional[bool] = None
):
    """
    设置日志记录系统（所有参数可选，未传入时从 config.settings 读取）

    Args:
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_format: 日志格式（仅在 use_json_format=False 时生效）
        date_format: 日期格式（仅在 use_json_format=False 时生效）
        log_file: 日志文件名
        log_dir: 日志目录
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        rotation_type: 轮转类型 ("size" 基于大小, "time" 基于时间)
        use_json_format: 是否使用 JSON 结构化日志（生产环境推荐）
    """
    from app.config import settings

    # 从 config.settings 读取默认值
    log_level = log_level or settings.LOG_LEVEL
    log_format = log_format or settings.LOG_FORMAT
    date_format = date_format or settings.LOG_DATE_FORMAT
    log_file = log_file or settings.LOG_FILE
    log_dir = log_dir or settings.LOG_DIR
    max_bytes = max_bytes if max_bytes is not None else settings.LOG_MAX_BYTES
    backup_count = backup_count if backup_count is not None else settings.LOG_BACKUP_COUNT
    enable_console = enable_console if enable_console is not None else settings.LOG_ENABLE_CONSOLE
    enable_file = enable_file if enable_file is not None else settings.LOG_ENABLE_FILE
    rotation_type = rotation_type or settings.LOG_ROTATION_TYPE
    use_json_format = use_json_format if use_json_format is not None else settings.LOG_FORMAT_JSON

    # 转换日志级别字符串为 logging 常量
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)

    # 清除现有的handlers避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # 创建格式化器（根据配置选择 JSON 或文本格式）
    if use_json_format:
        formatter: logging.Formatter = JsonFormatter()
    else:
        # 文本格式支持 TraceID（需在 log_format 中添加 %(trace_id)s）
        formatter = logging.Formatter(log_format, date_format)
        # 添加 TraceID 过滤器
        trace_filter = TraceIDFilter()
        root_logger.addFilter(trace_filter)

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(formatter)
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
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.info("日志系统初始化成功")
    root_logger.info(f"日志级别: {log_level}")
    root_logger.info(f"日志格式: {'JSON' if use_json_format else '文本'}")
    if enable_file:
        root_logger.info(f"日志文件: {log_file}")
        root_logger.info(f"轮转类型: {rotation_type}")
    if enable_console:
        root_logger.info("控制台输出: 启用")


logger = logging.getLogger(__name__)
