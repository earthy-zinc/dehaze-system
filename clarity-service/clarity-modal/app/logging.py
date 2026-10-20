"""日志（对齐日志架构设计 §2/§3）：logs/{yyyy-MM-dd}/{info,error}.log，NDJSON。

字段沿用三端契约：基础字段（timestamp/level/logger/service/thread/message）+ 请求上下文
（trace_id/method/path/ip/user_agent，自动注入）+ 场景字段（由调用点经 extra 提供，如 ACCESS
的 status/duration_ms、异常日志的 code/status）。文件按日期分目录、按大小分片，并清理过期目录。
"""

import logging
import shutil
import sys
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter as BaseJsonFormatter

from app.config import settings

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
_method_var: ContextVar[str] = ContextVar("method", default="")
_path_var: ContextVar[str] = ContextVar("path", default="")
_ip_var: ContextVar[str] = ContextVar("ip", default="")
_user_agent_var: ContextVar[str] = ContextVar("user_agent", default="")


def set_request_context(
    trace_id: str = "", method: str = "", path: str = "", ip: str = "", user_agent: str = ""
) -> None:
    """请求中间件在入口设置，JsonFormatter 自动注入该请求期间产生的每条日志"""
    _trace_id_var.set(trace_id)
    _method_var.set(method)
    _path_var.set(path)
    _ip_var.set(ip)
    _user_agent_var.set(user_agent)


class JsonFormatter(BaseJsonFormatter):
    def __init__(self, *args, **kwargs):
        # 关闭 ASCII 转义，中文按 UTF-8 原样输出，避免出现 \uXXXX
        kwargs.setdefault("json_ensure_ascii", False)
        super().__init__(*args, **kwargs)

    def add_fields(self, log_data: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_data, record, message_dict)
        log_data["timestamp"] = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["service"] = settings.SERVICE_NAME
        log_data["thread"] = record.thread
        log_data["trace_id"] = _trace_id_var.get()
        context = {
            "method": _method_var.get(),
            "path": _path_var.get(),
            "ip": _ip_var.get(),
            "user_agent": _user_agent_var.get(),
        }
        log_data.update({key: value for key, value in context.items() if value})
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)


class DateDirectoryFileHandler(logging.FileHandler):
    """按日期分目录 + 按大小分片的文件处理器（与三端 logs/{yyyy-MM-dd}/{级别}.log 结构对齐）"""

    def __init__(
        self, log_dir: str, filename: str, *, retention_days: int = 30, max_bytes: int = 0
    ) -> None:
        self._log_dir = Path(log_dir)
        self._filename = filename
        self._retention_days = retention_days
        self._max_bytes = max_bytes
        self._date = _today()
        Path(self._log_dir, self._date).mkdir(parents=True, exist_ok=True)
        super().__init__(
            str(self._log_dir / self._date / filename), mode="a", encoding="utf-8", delay=True
        )

    def emit(self, record: logging.LogRecord) -> None:
        today = _today()
        if today != self._date:
            self._rotate(today)
        super().emit(record)
        if self._max_bytes and self._size() >= self._max_bytes:
            self._archive()

    def _size(self) -> int:
        if not self.stream:
            return 0
        self.stream.flush()
        try:
            return Path(self.baseFilename).stat().st_size
        except OSError:
            return 0

    def _archive(self) -> None:
        """超限归档为 {级别}.{n}.log 并开新活动文件；归档失败则继续追加原文件（不丢日志）"""
        self.close()
        archived = self._next_archived_path()
        try:
            Path(self.baseFilename).replace(archived)
        except OSError:
            logging.getLogger(__name__).warning(
                "日志归档失败: %s", self.baseFilename, exc_info=True
            )
        self.stream = self._open()

    def _next_archived_path(self) -> Path:
        directory = Path(self.baseFilename).parent
        stem = Path(self._filename).stem
        numbers = [
            int(name)
            for path in directory.glob(f"{stem}.[0-9]*.log")
            if (name := path.name[len(stem) + 1 : -len(".log")]).isdigit()
        ]
        return directory / f"{stem}.{max(numbers, default=0) + 1}.log"

    def _rotate(self, date: str) -> None:
        self.close()
        self._date = date
        Path(self._log_dir, date).mkdir(parents=True, exist_ok=True)
        self.baseFilename = str((self._log_dir / date / self._filename).resolve())
        self.stream = self._open()
        self._clean_old_dirs()

    def _clean_old_dirs(self) -> None:
        if self._retention_days <= 0:
            return
        cutoff = (datetime.now() - timedelta(days=self._retention_days)).strftime("%Y-%m-%d")
        for entry in self._log_dir.iterdir():
            if entry.is_dir() and entry.name < cutoff:
                shutil.rmtree(entry, ignore_errors=True)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    root.addHandler(console)

    Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
    for level, filename in ((logging.INFO, "info.log"), (logging.ERROR, "error.log")):
        handler = DateDirectoryFileHandler(
            settings.LOG_DIR,
            filename,
            retention_days=settings.LOG_RETENTION_DAYS,
            max_bytes=settings.LOG_MAX_BYTES,
        )
        handler.setLevel(level)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.info("日志系统初始化成功 目录=%s/{yyyy-MM-dd}/info.log|error.log", settings.LOG_DIR)
