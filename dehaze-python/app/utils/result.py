from flask import jsonify, Response

from app.utils.code import ResultCode


def success(data=None, msg="success", code="00000"):
    """生成成功响应"""
    return jsonify({
        "code": code,
        "data": data,
        "msg": msg
    })


def error(msg: str, code: int = 500):
    """生成错误响应"""
    response = jsonify({
        "code": "B0001",
        "data": None,
        "msg": msg
    })
    response.status_code = code
    return response


def warning(code: ResultCode):
    return jsonify({
        "code": code.code,
        "data": None,
        "msg": code.msg
    })
