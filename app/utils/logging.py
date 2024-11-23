import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """设置日志记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler("app.log", maxBytes=10 * 1024 * 1024, backupCount=3)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
