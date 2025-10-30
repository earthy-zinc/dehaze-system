from flasgger import swag_from
from flask import Blueprint, request

from app.service.auth_service import AuthService
from app.utils.result import success, error

auth_blueprint = Blueprint('auth', __name__, url_prefix='/api/v1/auth')


@auth_blueprint.route('/login', methods=['POST'])
@swag_from({
    'tags': ['认证中心'],
    'summary': '登录',
    'description': '用户登录接口',
    'requestBody': {
        'content': {
            'application/json': {
                'schema': {
                    'type': 'object',
                    'properties': {
                        'username': {
                            'type': 'string',
                            'description': '用户名'
                        },
                        'password': {
                            'type': 'string',
                            'description': '密码'
                        }
                    }
                }
            }
        }
    },
    'responses': {
        '200': {
            'description': '登录成功',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string'},
                            'msg': {'type': 'string'},
                            'data': {
                                'type': 'object',
                                'properties': {
                                    'tokenType': {'type': 'string'},
                                    'accessToken': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            }
        },
        '401': {
            'description': '认证失败'
        }
    }
})
def login():
    """用户登录"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return error('用户名和密码不能为空', 400)

        result = AuthService.login(username, password)
        return success(result)
    except Exception as e:
        return error(str(e), 401)


@auth_blueprint.route('/logout', methods=['DELETE'])
@swag_from({
    'tags': ['认证中心'],
    'summary': '注销',
    'description': '用户注销接口',
    'security': [{'BearerAuth': []}],
    'responses': {
        '200': {
            'description': '注销成功'
        }
    }
})
def logout():
    """用户注销"""
    try:
        AuthService.logout()
        return success(None, '注销成功')
    except Exception as e:
        return error(str(e), 400)


@auth_blueprint.route('/captcha', methods=['GET'])
@swag_from({
    'tags': ['认证中心'],
    'summary': '获取验证码',
    'description': '获取验证码图片',
    'responses': {
        '200': {
            'description': '获取成功',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string'},
                            'msg': {'type': 'string'},
                            'data': {
                                'type': 'object',
                                'properties': {
                                    'captchaKey': {'type': 'string'},
                                    'captchaBase64': {'type': 'string'}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
})
def get_captcha():
    """获取验证码"""
    try:
        result = AuthService.get_captcha()
        return success(result)
    except Exception as e:
        return error(str(e), 500)
