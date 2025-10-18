from flask import Blueprint, current_app
from flask_socketio import emit
from app.utils.jwt_util import jwt_required, get_current_user_id
from flasgger import swag_from

# 注意：WebSocket路由通常不使用Blueprint，这里仅用于文档说明
# 实际的WebSocket事件处理在websocket_service.py中实现

websocket_blueprint = Blueprint('websocket', __name__, url_prefix='/websocket')


@websocket_blueprint.route('/docs', methods=['GET'])
@swag_from({
    'tags': ['WebSocket'],
    'summary': 'WebSocket接口文档',
    'description': 'WebSocket通信接口说明',
    'responses': {
        '200': {
            'description': '接口文档',
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'message': {'type': 'string'}
                        }
                    }
                }
            }
        }
    }
})
def websocket_docs():
    """WebSocket接口文档"""
    return {
        'message': 'WebSocket接口文档',
        'events': {
            'connect': '客户端连接事件，需要JWT认证',
            'disconnect': '客户端断开连接事件',
            'send_to_all': '向所有连接的客户端广播消息',
            'send_to_user': '向指定用户发送私信'
        }
    }