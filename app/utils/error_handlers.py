from flask import Flask
from flask_jwt_extended.exceptions import NoAuthorizationError, InvalidHeaderError
from jwt import ExpiredSignatureError, DecodeError

from .result import error


def register_error_handlers(app: Flask):
    """注册全局错误处理"""

    @app.errorhandler(AssertionError)
    def handle_assertion_error(e):
        return error(f"业务逻辑错误：{str(e)}")

    @app.errorhandler(DecodeError)
    @app.errorhandler(InvalidHeaderError)
    def handle_invalid_header_error(e):
        return error(f"请求头错误：{str(e)}")

    @app.errorhandler(NoAuthorizationError)
    def handle_no_authorization_error(e):
        return error(f"未登录：{str(e)}")

    @app.errorhandler(ExpiredSignatureError)
    def handle_expired_signature_error(e):
        return error(f"登录过期：{str(e)}")

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        traceback.print_exc()
        return error(f"系统内部错误：{str(e)}")
