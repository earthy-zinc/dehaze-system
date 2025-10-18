from datetime import datetime
from flask import jsonify


def format_time(dt):
    """格式化时间"""
    if dt:
        if isinstance(dt, datetime):
            return dt.isoformat()
    return None


def result_util(code, msg, data):
    """统一返回结果格式"""
    return {
        'code': code,
        'msg': msg,
        'data': data
    }