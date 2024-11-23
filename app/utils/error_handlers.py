from flask import Flask

from .result import error


def register_error_handlers(app: Flask):
    """注册全局错误处理"""

    @app.errorhandler(AssertionError)
    def handle_assertion_error(e):
        return error(f"业务逻辑错误：{str(e)}")

    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback
        traceback.print_exc()
        return error(f"系统内部错误：{str(e)}")
