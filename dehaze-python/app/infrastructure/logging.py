import logging
import os
import shutil
import sys
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

from pythonjsonlogger.json import JsonFormatter as BaseJsonFormatter

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


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


class DailyDirectoryFileHandler(logging.FileHandler):
    """按日期分目录的文件日志处理器：{log_dir}/{yyyy-MM-dd}/{filename}

    当天日志写入 {log_dir}/{今天}/{filename}，跨天自动切换到新日期目录，
    并清理超过保留期的日期目录。与 Go/Java 端 logs/{yyyy-MM-dd}/{级别}.log 结构对齐。
    """

    def __init__(self, log_dir, filename, retention_days=30, encoding='utf-8'):
        self.log_dir = log_dir
        self.filename = filename
        self.retention_days = retention_days
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        super().__init__(self._path_for_date(self.current_date), mode='a',
                         encoding=encoding, delay=True)

    def _path_for_date(self, date_str):
        return os.path.join(self.log_dir, date_str, self.filename)

    def emit(self, record):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            self._rotate(today)
        super().emit(record)

    def _rotate(self, new_date):
        if self.stream:
            self.close()
        self.current_date = new_date
        self.baseFilename = os.path.abspath(self._path_for_date(new_date))
        self.stream = self._open()
        self._clean_old_logs()

    def _clean_old_logs(self):
        if self.retention_days <= 0 or not os.path.isdir(self.log_dir):
            return
        cutoff = (datetime.now() - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        for entry in os.scandir(self.log_dir):
            if entry.is_dir() and entry.name < cutoff:
                shutil.rmtree(entry.path, ignore_errors=True)


def setup_logging(use_json_format: Optional[bool] = None):
    """
    设置日志记录系统

    所有配置从 config.settings 读取，仅 use_json_format 可外部覆盖（用于测试）。
    文件输出统一为 logs/{yyyy-MM-dd}/{级别}.log，按日期分目录，info/error 分文件。
    """
    from app.config import settings

    log_level = settings.LOG_LEVEL
    log_format = settings.LOG_FORMAT
    date_format = settings.LOG_DATE_FORMAT
    log_dir = settings.LOG_DIR
    retention_days = settings.LOG_RETENTION_DAYS
    enable_console = settings.LOG_ENABLE_CONSOLE
    enable_file = settings.LOG_ENABLE_FILE
    use_json_format = use_json_format if use_json_format is not None else settings.LOG_FORMAT_JSON

    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    trace_filter = TraceIDFilter()

    file_formatter: logging.Formatter = JsonFormatter()
    if use_json_format:
        console_formatter: logging.Formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(log_format, date_format)

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_int)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(trace_filter)
        root_logger.addHandler(console_handler)

    if enable_file:
        os.makedirs(log_dir, exist_ok=True)

        info_handler = DailyDirectoryFileHandler(
            log_dir, "info.log", retention_days=retention_days
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(file_formatter)
        info_handler.addFilter(trace_filter)
        root_logger.addHandler(info_handler)

        error_handler = DailyDirectoryFileHandler(
            log_dir, "error.log", retention_days=retention_days
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        error_handler.addFilter(trace_filter)
        root_logger.addHandler(error_handler)

    root_logger.info("日志系统初始化成功")
    root_logger.info(f"日志级别: {log_level}")
    root_logger.info(f"控制台格式: {'JSON' if use_json_format else '文本'}")
    root_logger.info("文件格式: JSON")
    if enable_file:
        root_logger.info(f"日志目录: {log_dir}/{{yyyy-MM-dd}}/info.log|error.log")
        root_logger.info(f"保留天数: {retention_days}")
    if enable_console:
        root_logger.info("控制台输出: 启用")


logger = logging.getLogger(__name__)
