import os
import traceback

from flasgger import Swagger
from flask import Flask

from common import error
from config import config


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    from base import base as base_blueprint
    from base import db

    db.init_app(app)
    app.register_blueprint(base_blueprint)
    swagger = Swagger(app)
    return app, swagger


app, swagger = create_app(os.getenv('FLASK_CONFIG') or 'default')


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
