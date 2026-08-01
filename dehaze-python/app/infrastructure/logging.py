import logging
import os
import shutil
import sys
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Optional

from pythonjsonlogger.json import JsonFormatter as BaseJsonFormatter

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
_request_method_var: ContextVar[str] = ContextVar("request_method", default="")
_request_path_var: ContextVar[str] = ContextVar("request_path", default="")
_request_ip_var: ContextVar[str] = ContextVar("request_ip", default="")
_request_user_agent_var: ContextVar[str] = ContextVar("request_user_agent", default="")


def set_request_context(trace_id: str = "", method: str = "", path: str = "",
                        ip: str = "", user_agent: str = "") -> None:
    """请求中间件在入口设置当前请求上下文（traceId/method/path/ip/userAgent）。

    JsonFormatter 会把这些值自动注入到该请求期间产生的每条日志，使任意 logger
    输出都能定位到具体接口与来源，无需在各业务点手动拼接。user_id 由认证层通过
    app.models.base._current_user_id 注入，格式化器自动读取。
    """
    if trace_id:
        _trace_id_var.set(trace_id)
    if method:
        _request_method_var.set(method)
    if path:
        _request_path_var.set(path)
    if ip:
        _request_ip_var.set(ip)
    if user_agent:
        _request_user_agent_var.set(user_agent)


class JsonFormatter(BaseJsonFormatter):
    """JSON 结构化日志格式化器（支持请求上下文注入）"""

    def __init__(self, *args, **kwargs):
        # 关闭 ASCII 转义，中文按 UTF-8 原样输出，避免出现 \uXXXX
        kwargs.setdefault("json_ensure_ascii", False)
        super().__init__(*args, **kwargs)

    def add_fields(self, log_data, record, message_dict):
        super().add_fields(log_data, record, message_dict)
        from app.config import settings
        log_data["timestamp"] = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["service"] = settings.APP_NAME
        log_data["thread"] = record.thread
        log_data["trace_id"] = _trace_id_var.get("")
        method = _request_method_var.get("")
        path = _request_path_var.get("")
        if method:
            log_data["method"] = method
        if path:
            log_data["path"] = path
        ip = _request_ip_var.get("")
        if ip:
            log_data["ip"] = ip
        ua = _request_user_agent_var.get("")
        if ua:
            log_data["user_agent"] = ua
        # user_id 由认证层写入 _current_user_id（app.models.base），请求内日志自动带上
        from app.models.base import _current_user_id
        uid = _current_user_id.get()
        if uid is not None:
            log_data["user_id"] = uid


class TraceIDFilter(logging.Filter):
    """TraceID 注入过滤器（用于文本格式日志）"""

    def filter(self, record):
        record.trace_id = _trace_id_var.get("")
        return True


class DailyDirectoryFileHandler(logging.FileHandler):
    """按日期分目录 + 按大小分片的文件日志处理器。

    当天日志写入 {log_dir}/{今天}/{filename}，跨天切换到新日期目录；
    单文件超过 max_bytes 时，归档为 {stem}.{n}.log 并开新活动文件。
    与 Go/Java 端 logs/{yyyy-MM-dd}/{级别}.log 结构对齐。
    """

    def __init__(self, log_dir, filename, retention_days=30, max_bytes=0, encoding='utf-8'):
        self.log_dir = log_dir
        self.filename = filename
        self.retention_days = retention_days
        self.max_bytes = max_bytes
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.log_dir, self.current_date), exist_ok=True)
        super().__init__(self._path_for_date(self.current_date), mode='a',
                         encoding=encoding, delay=True)

    def _path_for_date(self, date_str):
        return os.path.join(self.log_dir, date_str, self.filename)

    def emit(self, record):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.current_date:
            self._rotate(today)
        super().emit(record)
        if self.max_bytes and self._exceeds_size():
            self._archive()

    def _exceeds_size(self):
        if self.stream is None:
            return False
        self.stream.flush()
        try:
            return os.path.getsize(self.baseFilename) >= self.max_bytes
        except OSError:
            return False

    def _archive(self):
        if self.stream:
            self.close()
        archived = self._next_archived_path()
        renamed = False
        try:
            os.rename(self.baseFilename, archived)
            renamed = True
        except OSError:
            # 归档失败则继续追加原文件，避免丢日志
            pass
        try:
            self.stream = self._open()
        except OSError:
            if renamed:
                # 新文件打开失败，尝试回滚重命名避免丢日志
                try:
                    os.rename(archived, self.baseFilename)
                except OSError:
                    pass
            self.stream = self._open()  # 再次尝试打开原路径

    def archive_existing(self):
        """启动时归档当天已存在的活动文件（dev 用），需在文件打开前调用。"""
        if self.stream is not None:
            return
        if not os.path.exists(self.baseFilename):
            return
        try:
            if os.path.getsize(self.baseFilename) == 0:
                return
        except OSError:
            return
        archived = self._next_archived_path()
        try:
            os.rename(self.baseFilename, archived)
        except OSError:
            pass

    def _next_archived_path(self):
        dir_ = os.path.dirname(self.baseFilename)
        stem = os.path.splitext(self.filename)[0]  # info / error
        prefix = f"{stem}."
        n = 0
        try:
            for entry in os.scandir(dir_):
                if not entry.is_file():
                    continue
                name = entry.name
                if name.startswith(prefix) and name.endswith(".log"):
                    num = name[len(prefix):-len(".log")]
                    if num.isdigit():
                        n = max(n, int(num))
        except OSError:
            pass
        return os.path.join(dir_, f"{stem}.{n + 1}.log")

    def _rotate(self, new_date):
        if self.stream:
            self.close()
        self.current_date = new_date
        os.makedirs(os.path.join(self.log_dir, new_date), exist_ok=True)
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
    log_max_bytes = settings.LOG_MAX_BYTES
    log_archive_on_startup = settings.LOG_ARCHIVE_ON_STARTUP
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
            log_dir, "info.log", retention_days=retention_days,
            max_bytes=log_max_bytes
        )
        if log_archive_on_startup:
            info_handler.archive_existing()
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(file_formatter)
        info_handler.addFilter(trace_filter)
        root_logger.addHandler(info_handler)

        error_handler = DailyDirectoryFileHandler(
            log_dir, "error.log", retention_days=retention_days,
            max_bytes=log_max_bytes
        )
        if log_archive_on_startup:
            error_handler.archive_existing()
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        error_handler.addFilter(trace_filter)
        root_logger.addHandler(error_handler)

    # SQL 审计日志级别独立控制：开发/测试记录全部 SQL，生产仅记慢查询与错误
    logging.getLogger("sql").setLevel(
        getattr(logging, settings.SQL_LOG_LEVEL.upper(), logging.INFO)
    )

    root_logger.info("日志系统初始化成功")
    root_logger.info(f"日志级别: {log_level}")
    root_logger.info(f"SQL 审计日志级别: {settings.SQL_LOG_LEVEL}")
    root_logger.info(f"控制台格式: {'JSON' if use_json_format else '文本'}")
    root_logger.info("文件格式: JSON")
    if enable_file:
        root_logger.info(f"日志目录: {log_dir}/{{yyyy-MM-dd}}/info.log|error.log")
        root_logger.info(f"保留天数: {retention_days}")
    if enable_console:
        root_logger.info("控制台输出: 启用")


logger = logging.getLogger(__name__)
