import unittest
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.service.websocket_service import WebSocketService


class TestWebSocketService(unittest.TestCase):

    def setUp(self):
        """测试前准备"""
        self.mock_socketio = MagicMock()

    def test_websocket_service_initialization(self):
        """测试WebSocket服务初始化"""
        with patch('app.service.websocket_service.join_room'), \
             patch('app.service.websocket_service.leave_room'), \
             patch('app.service.websocket_service.emit'):
            websocket_service = WebSocketService(self.mock_socketio)
            # 验证事件处理函数被正确注册
            self.mock_socketio.on.assert_called()

    def test_websocket_service_events_registered(self):
        """测试WebSocket事件已注册"""
        with patch('app.service.websocket_service.join_room'), \
             patch('app.service.websocket_service.leave_room'), \
             patch('app.service.websocket_service.emit'):
            websocket_service = WebSocketService(self.mock_socketio)
            # 验证所有事件都被注册
            calls = [call[0][0] for call in self.mock_socketio.on.call_args_list]
            self.assertIn('connect', calls)
            self.assertIn('disconnect', calls)
            self.assertIn('send_to_all', calls)
            self.assertIn('send_to_user', calls)


if __name__ == '__main__':
    unittest.main()