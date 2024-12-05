from flask import jsonify

from app.utils.code import ResultCode


def success(data):
    return jsonify({
        "code": "00000",
        "data": data,
        "msg": "success"
    })

def error(msg: str, response_code: int=500):
    return jsonify({
        "code": ResultCode.SYSTEM_EXECUTION_ERROR,
        "data": None,
        "msg": msg
    }), response_code

def warning(code: ResultCode):
    return jsonify({
        "code": code.code,
        "data": None,
        "msg": code.msg
    })
