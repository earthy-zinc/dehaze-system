def init_routes(app):
    from app.route.model import model_blueprint
    from app.route.user import user_blueprint
    from app.route.role import role_blueprint
    from app.route.menu import menu_blueprint
    from app.route.dict import dict_blueprint
    from app.route.dataset import dataset_blueprint
    from app.route.algorithm import algorithm_blueprint
    from app.route.dept import dept_blueprint
    from app.route.file import file_blueprint
    from app.route.item_file import item_file_blueprint
    from app.route.auth import auth_blueprint
    from app.route.websocket import websocket_blueprint

    app.register_blueprint(model_blueprint)
    app.register_blueprint(user_blueprint)
    app.register_blueprint(role_blueprint)
    app.register_blueprint(menu_blueprint)
    app.register_blueprint(dict_blueprint)
    app.register_blueprint(dataset_blueprint)
    app.register_blueprint(algorithm_blueprint)
    app.register_blueprint(dept_blueprint)
    app.register_blueprint(file_blueprint)
    app.register_blueprint(item_file_blueprint)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(websocket_blueprint)
