import os

from app import create_app

app = create_app(os.getenv("FLASK_ENV", "development"))

if __name__ == "__main__":
    # Flask-SocketIO 需要使用 socketio.run() 而不是 app.run()
    # 获取在 create_app 中初始化的 socketio 实例
    from flask_socketio import SocketIO
    from flask import current_app

    # 由于 create_app 没有返回 socketio，我们需要重新获取
    # 使用全局 socketio 运行器
    socketio = SocketIO(cors_allowed_origins="*")
    socketio.init_app(app, async_mode='threading')
    # allow_unsafe_werkzeug=True 用于开发环境，生产环境应使用 eventlet/gevent
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
