def init_routes(app):
    from app.route.model import model_blueprint
    from app.route.user import user_blueprint
    from app.route.role import role_blueprint
    from app.route.menu import menu_blueprint
    from app.route.dict import dict_blueprint

    app.register_blueprint(model_blueprint)
    app.register_blueprint(user_blueprint)
    app.register_blueprint(role_blueprint)
    app.register_blueprint(menu_blueprint)
    app.register_blueprint(dict_blueprint)
