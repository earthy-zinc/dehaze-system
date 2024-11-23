import os
from app import create_app

if __name__ == '__main__':
    # 获取环境配置并创建 app
    app = create_app(os.getenv("FLASK_ENV", "default"))
    app.run(host='0.0.0.0', port=5000)
