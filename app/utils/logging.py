import logging
import logging.config
import sys
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional

# 默认配置
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s --- [%(thread)d] %(name)s : %(message)s"
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

class UTF8RotatingFileHandler(RotatingFileHandler):
    """支持 UTF-8 编码的轮转文件处理器"""
    def __init__(self, filename, mode='a', max_bytes=0, backup_count=0, encoding='utf-8', delay=False):
        # 确保日志目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        super().__init__(filename, mode, max_bytes, backup_count, encoding, delay)

class UTF8TimedRotatingFileHandler(TimedRotatingFileHandler):
    """支持 UTF-8 编码的定时轮转文件处理器"""
    def __init__(self, filename, when='midnight', interval=1, backup_count=0,
                 encoding='utf-8', delay=False, utc=False, at_time=None):
        # 确保日志目录存在
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        super().__init__(filename, when, interval, backup_count, encoding, delay, utc, at_time)

def setup_logging(
        log_level: int = DEFAULT_LOG_LEVEL,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT,
        log_file: Optional[str] = None,
        log_dir: str = "logs",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        enable_console: bool = True,
        enable_file: bool = True,
        rotation_type: str = "size"  # "size" 或 "time"
):
    """
    设置日志记录系统

    Args:
        log_level: 日志级别
        log_format: 日志格式
        date_format: 日期格式
        log_file: 日志文件名 (默认为 app.log)
        log_dir: 日志目录
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        enable_console: 是否启用控制台输出
        enable_file: 是否启用文件输出
        rotation_type: 轮转类型 ("size" 基于大小, "time" 基于时间)
    """
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除现有的handlers避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # 创建格式化器
    formatter = logging.Formatter(log_format, date_format)

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
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
            file_handler = UTF8TimedRotatingFileHandler(
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

        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 记录日志系统初始化
    root_logger.info("日志系统初始化成功")
    root_logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    if enable_file:
        root_logger.info(f"日志文件: {log_file}")
        root_logger.info(f"轮转类型: {rotation_type}")
    if enable_console:
        root_logger.info("控制台输出: 启用")
