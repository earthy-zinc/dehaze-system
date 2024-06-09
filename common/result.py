from flask import jsonify


def success(data):
    return jsonify({
        "code": 0,
        "data": data,
        "msg": "success"
    })
    
def error(msg):
    return jsonify({
        "code": 1,
        "data": None,
        "msg": msg
    })
    
def warning(msg):
    return jsonify({
        "code": 2,
        "data": None,
        "msg": msg
    })
    
def info(data, msg):
    return jsonify({
        "code": 3,
        "data": data,
        "msg": msg
    })
