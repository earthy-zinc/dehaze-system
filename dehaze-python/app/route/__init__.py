from flask import Blueprint


def init_routes(app):
    from app.route.model import model_blueprint
    from app.route.user import user_blueprint

    app.register_blueprint(model_blueprint)
    app.register_blueprint(user_blueprint)