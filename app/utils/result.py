from flask import jsonify


def success(data):
    return jsonify({
        "code": "00000",
        "data": data,
        "msg": "success"
    })

def error(msg, response_code=500):
    return jsonify({
        "code": "B00001",
        "data": None,
        "msg": msg
    }), response_code
