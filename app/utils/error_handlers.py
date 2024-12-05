from flask import Flask
from flask_jwt_extended.exceptions import NoAuthorizationError, InvalidHeaderError
from jwt import ExpiredSignatureError, DecodeError

from .code import ResultCode
from .result import error, warning


def register_error_handlers(app: Flask):
    """注册全局错误处理"""

    @app.errorhandler(AssertionError)
    def handle_assertion_error(e):
        return error(f"业务逻辑错误：{str(e)}"), 400

    @app.errorhandler(NoAuthorizationError)
    def handle_no_authorization_error(e):
        return warning(ResultCode.ACCESS_UNAUTHORIZED), 401

    @app.errorhandler(DecodeError)
    @app.errorhandler(InvalidHeaderError)
    @app.errorhandler(ExpiredSignatureError)
    def handle_jwt_error(e):
        return warning(ResultCode.TOKEN_INVALID), 401

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        traceback.print_exc()
        return error(f"系统内部错误：{str(e)}"), 500
