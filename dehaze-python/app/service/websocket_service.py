from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
from app.utils.jwt_util import jwt_required, get_current_user_id
import json


class WebSocketService:
    """WebSocket服务类"""

    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.register_events()

    def register_events(self):
        """注册WebSocket事件处理函数"""
        
        @self.socketio.on('connect')
        def handle_connect():
            # 用户连接时的处理
            user_id = get_current_user_id()
            if user_id:
                join_room(f"user_{user_id}")
                emit('connected', {'message': f'User {user_id} connected successfully'})
            else:
                emit('error', {'message': 'Authentication required'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            # 用户断开连接时的处理
            user_id = get_current_user_id()
            if user_id:
                leave_room(f"user_{user_id}")
                
        @self.socketio.on('send_to_all')
        def handle_send_to_all(data):
            # 广播消息给所有用户
            user_id = get_current_user_id()
            if user_id:
                message = f"User {user_id}: {data}"
                self.socketio.emit('broadcast', {'message': message}, room=None)
            else:
                emit('error', {'message': 'Authentication required'})
                
        @self.socketio.on('send_to_user')
        def handle_send_to_user(data):
            # 发送消息给特定用户
            user_id = get_current_user_id()
            if user_id:
                target_user_id = data.get('target_user_id')
                message = data.get('message')
                if target_user_id and message:
                    target_room = f"user_{target_user_id}"
                    self.socketio.emit('private_message', {
                        'sender_id': user_id,
                        'message': message
                    }, room=target_room)
                else:
                    emit('error', {'message': 'Invalid data'})
            else:
                emit('error', {'message': 'Authentication required'})