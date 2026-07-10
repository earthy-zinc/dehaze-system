from datetime import datetime


def format_time(dt):
    """格式化时间为 ISO 8601 字符串"""
    if dt:
        if isinstance(dt, datetime):
            return dt.isoformat()
    return None
