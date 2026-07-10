"""
WebSocket 服务

使用 FastAPI 原生 WebSocket 实现实时通信
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from fastapi import WebSocket, WebSocketDisconnect
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket 连接信息"""
    websocket: WebSocket
    user_id: int
    username: str = ""

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
        }


class ConnectionManager:
    """
    WebSocket 连接管理器

    管理所有活跃的 WebSocket 连接，支持：
    - 用户连接/断开
    - 广播消息
    - 定向推送
    """

    def __init__(self):
        # user_id -> list[WebSocketConnection]
        self._connections: dict[int, list[WebSocketConnection]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: int, username: str = "") -> bool:
        """
        接受 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
            user_id: 用户 ID
            username: 用户名

        Returns:
            是否连接成功
        """
        try:
            await websocket.accept()
            conn = WebSocketConnection(
                websocket=websocket,
                user_id=user_id,
                username=username,
            )

            async with self._lock:
                if user_id not in self._connections:
                    self._connections[user_id] = []
                self._connections[user_id].append(conn)

            logger.info(
                f"WebSocket 连接成功: user_id={user_id}, username={username}")
            return True
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            return False

    async def disconnect(self, websocket: WebSocket, user_id: int):
        """
        断开 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
            user_id: 用户 ID
        """
        async with self._lock:
            if user_id in self._connections:
                self._connections[user_id] = [
                    conn for conn in self._connections[user_id]
                    if conn.websocket != websocket
                ]
                if not self._connections[user_id]:
                    del self._connections[user_id]

        logger.info(f"WebSocket 断开连接: user_id={user_id}")

    async def send_personal(self, user_id: int, message: dict[str, Any]):
        """
        向指定用户发送消息（所有连接）

        Args:
            user_id: 用户 ID
            message: 消息内容
        """
        async with self._lock:
            connections = self._connections.get(user_id, [])[:]

        if not connections:
            return

        message_json = json.dumps(message, ensure_ascii=False)
        disconnected = []

        for conn in connections:
            try:
                await conn.websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"发送消息失败: user_id={user_id}, error={e}")
                disconnected.append(conn.websocket)

        # 清理断开的连接
        for ws in disconnected:
            await self.disconnect(ws, user_id)

    async def broadcast(self, message: dict[str, Any], exclude_user: int | None = None):
        """
        广播消息给所有用户

        Args:
            message: 消息内容
            exclude_user: 排除的用户 ID
        """
        async with self._lock:
            all_connections: list[WebSocketConnection] = []
            for user_id, conns in self._connections.items():
                if user_id != exclude_user:
                    all_connections.extend(conns)

        if not all_connections:
            return

        message_json = json.dumps(message, ensure_ascii=False)

        for conn in all_connections:
            try:
                await conn.websocket.send_text(message_json)
            except Exception as e:
                logger.warning(f"广播消息失败: user_id={conn.user_id}, error={e}")

    async def get_online_users(self) -> list[dict[str, Any]]:
        """
        获取在线用户列表

        Returns:
            在线用户信息列表
        """
        async with self._lock:
            users = []
            seen = set()
            for user_id, connections in self._connections.items():
                if user_id not in seen and connections:
                    seen.add(user_id)
                    users.append(connections[0].to_dict())
            return users

    @property
    def online_count(self) -> int:
        """在线用户数"""
        return len(self._connections)


# 全局连接管理器
manager = ConnectionManager()


class WebSocketService:
    """WebSocket 服务类（异步版本）"""

    @staticmethod
    async def verify_token(token: str) -> dict[str, Any] | None:
        """
        验证 JWT Token

        Args:
            token: JWT Token 字符串

        Returns:
            解码后的 payload，验证失败返回 None
        """
        if not token:
            return None

        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=["HS256"]
            )
            return payload
        except ExpiredSignatureError:
            logger.warning("Token 已过期")
            return None
        except JWTError as e:
            logger.warning(f"Token 验证失败: {e}")
            return None

    @staticmethod
    async def broadcast_shutdown_notification():
        """
        广播服务器关闭通知

        通知所有连接的客户端服务器即将关闭
        """
        try:
            await manager.broadcast({
                "type": "server_shutdown",
                "message": "服务器正在关闭，请保存工作",
                "reconnect": False,
            })
            logger.info(f"已广播关闭通知给 {manager.online_count} 个用户")
        except Exception as e:
            logger.error(f"广播关闭通知失败: {e}")

    @staticmethod
    async def handle_connection(websocket: WebSocket, token: str):
        """
        处理 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
            token: JWT Token
        """
        # 验证 Token
        payload = await WebSocketService.verify_token(token)
        if not payload:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "message": "认证失败，请重新登录"
            })
            await websocket.close(code=4001)
            return

        user_id = payload.get("user_id")
        username = payload.get("username", "")

        if not user_id:
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "message": "无效的用户信息"
            })
            await websocket.close(code=4002)
            return

        # 建立连接
        connected = await manager.connect(websocket, user_id, username)
        if not connected:
            return

        try:
            # 发送连接成功消息
            await websocket.send_json({
                "type": "connected",
                "message": f"用户 {username} 连接成功",
                "user_id": user_id,
            })

            # 通知其他用户
            await manager.broadcast({
                "type": "user_online",
                "user_id": user_id,
                "username": username,
            }, exclude_user=user_id)

            # 持续监听消息
            while True:
                data = await websocket.receive_text()
                await WebSocketService._handle_message(websocket, user_id, username, data)

        except WebSocketDisconnect:
            logger.info(f"用户断开连接: user_id={user_id}")
        except Exception as e:
            logger.error(f"WebSocket 异常: user_id={user_id}, error={e}")
        finally:
            await manager.disconnect(websocket, user_id)
            # 通知其他用户
            await manager.broadcast({
                "type": "user_offline",
                "user_id": user_id,
                "username": username,
            })

    @staticmethod
    async def _handle_message(
        websocket: WebSocket,
        user_id: int,
        username: str,
        data: str
    ):
        """
        处理 WebSocket 消息

        Args:
            websocket: WebSocket 连接对象
            user_id: 用户 ID
            username: 用户名
            data: 消息数据
        """
        try:
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "broadcast":
                content = message.get("message", "")
                await manager.broadcast({
                    "type": "broadcast",
                    "user_id": user_id,
                    "username": username,
                    "message": content,
                })

            elif msg_type == "private":
                target_user_id = message.get("target_user_id")
                content = message.get("message", "")
                if target_user_id:
                    await manager.send_personal(target_user_id, {
                        "type": "private_message",
                        "sender_id": user_id,
                        "sender_name": username,
                        "message": content,
                    })

            elif msg_type == "get_online_users":
                users = await manager.get_online_users()
                await websocket.send_json({
                    "type": "online_users",
                    "users": users,
                    "count": len(users),
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}"
                })

        except json.JSONDecodeError:
            await websocket.send_json({
                "type": "error",
                "message": "无效的 JSON 格式"
            })
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "处理消息失败"
            })
