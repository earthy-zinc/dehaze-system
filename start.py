from http.client import HTTPException
import os
import traceback
from flask import Flask
from config import config
from common import error

def create_app(config_name):
    app = Flask(__name__)
    
    app.config.from_object(config[config_name])

    from base import base as base_blueprint
    app.register_blueprint(base_blueprint)
    return app


app = create_app(os.getenv('FLASK_CONFIG') or 'default')


@app.errorhandler(AssertionError)
def handle_service_error(e):
    return error(str(e))

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return error(str(e))


if __name__ == '__main__':
    if os.getenv('FLASK_CONFIG') != 'production':
        app.debug = True
    app.run(host='0.0.0.0', port=5000)