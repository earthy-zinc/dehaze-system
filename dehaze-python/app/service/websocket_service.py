"""
WebSocket 服务（跨 Worker 支持）

使用 Redis Pub/Sub 实现多 Worker 间的消息广播：
- 每个 Worker 维护本地 WebSocket 连接
- send_personal / broadcast 通过 Redis Pub/Sub 跨 Worker 投递
- 在线用户列表通过 Redis sorted set + 心跳维护

降级策略：Redis 不可用时自动降级为本地单 Worker 模式
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from app.config import settings
from app.dependencies.redis import get_redis_client
from fastapi import WebSocket, WebSocketDisconnect
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError
from redis.asyncio import Redis

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


class DistributedConnectionManager:
    """
    跨 Worker 的 WebSocket 连接管理器

    - 本地连接：dict[user_id, list[WebSocketConnection]]，仅本 Worker 的连接
    - 跨 Worker 通信：Redis Pub/Sub 频道 dehaze:ws:broadcast
    - 在线用户：Redis sorted set dehaze:ws:online_users，score=心跳时间戳
    """

    def __init__(self):
        self._local_connections: dict[int, list[WebSocketConnection]] = {}
        self._lock = asyncio.Lock()
        self._redis: Optional[Redis] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._started = False

    async def start(self, redis: Redis):
        """启动跨 Worker 通信（在 lifespan 中调用）"""
        if self._started:
            return
        self._redis = redis
        self._started = True
        self._pubsub_task = asyncio.create_task(self._subscribe_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket 跨 Worker 通信已启动")

    async def stop(self):
        """停止跨 Worker 通信（在 lifespan 中调用）"""
        self._started = False
        for task in (self._pubsub_task, self._heartbeat_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._pubsub_task = None
        self._heartbeat_task = None
        logger.info("WebSocket 跨 Worker 通信已停止")

    # ===== 连接管理 =====

    async def connect(self, websocket: WebSocket, user_id: int, username: str = "") -> bool:
        """接受 WebSocket 连接"""
        try:
            await websocket.accept()
            conn = WebSocketConnection(
                websocket=websocket,
                user_id=user_id,
                username=username,
            )

            async with self._lock:
                if user_id not in self._local_connections:
                    self._local_connections[user_id] = []
                self._local_connections[user_id].append(conn)

            # 更新 Redis 在线状态
            await self._update_online_status(user_id)

            logger.info(f"WebSocket 连接成功: user_id={user_id}, username={username}")
            return True
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            return False

    async def disconnect(self, websocket: WebSocket, user_id: int):
        """断开 WebSocket 连接"""
        await self._disconnect_local(websocket, user_id)

        # 如果本 Worker 已无该用户的连接，由心跳机制负责清理 Redis 在线状态
        async with self._lock:
            has_local = user_id in self._local_connections and self._local_connections[user_id]

        if not has_local:
            # 主动从 Redis 清除（延迟，防止重连瞬间被清）
            asyncio.create_task(self._delayed_cleanup_online(user_id))

        logger.info(f"WebSocket 断开连接: user_id={user_id}")

    async def _disconnect_local(self, websocket: WebSocket, user_id: int):
        """清理本地连接"""
        async with self._lock:
            if user_id in self._local_connections:
                self._local_connections[user_id] = [
                    conn for conn in self._local_connections[user_id]
                    if conn.websocket != websocket
                ]
                if not self._local_connections[user_id]:
                    del self._local_connections[user_id]

    async def _delayed_cleanup_online(self, user_id: int):
        """延迟清理 Redis 在线状态（5 秒后，防止快速重连被误清）"""
        await asyncio.sleep(5)
        async with self._lock:
            has_local = user_id in self._local_connections and self._local_connections[user_id]
        if not has_local and self._redis:
            try:
                await self._redis.zrem(settings.WS_ONLINE_KEY, str(user_id))
            except Exception as e:
                logger.warning(f"清理 Redis 在线状态失败: user_id={user_id}, error={e}")

    # ===== 消息投递 =====

    async def send_personal(self, user_id: int, message: dict[str, Any]):
        """向指定用户发送消息（跨 Worker）"""
        if self._redis and self._started:
            try:
                msg = json.dumps({
                    "target_user_id": user_id,
                    "message": message,
                }, ensure_ascii=False)
                await self._redis.publish(settings.WS_REDIS_CHANNEL, msg)
                return
            except Exception as e:
                logger.warning(f"Redis Pub/Sub 发布失败，降级为本地发送: {e}")

        # 降级：仅本地发送
        await self._send_to_local_user(user_id, message)

    async def broadcast(self, message: dict[str, Any], exclude_user: int | None = None):
        """广播消息（跨 Worker）"""
        if self._redis and self._started:
            try:
                msg = json.dumps({
                    "exclude_user": exclude_user,
                    "message": message,
                }, ensure_ascii=False)
                await self._redis.publish(settings.WS_REDIS_CHANNEL, msg)
                return
            except Exception as e:
                logger.warning(f"Redis Pub/Sub 广播失败，降级为本地广播: {e}")

        await self._broadcast_local(message, exclude_user)

    async def get_online_users(self) -> list[dict[str, Any]]:
        """获取在线用户列表（跨 Worker）"""
        if self._redis and self._started:
            try:
                now = time.time()
                min_score = now - settings.WS_ONLINE_TTL
                raw = await self._redis.zrangebyscore(
                    settings.WS_ONLINE_KEY, min_score, now
                )
                # 从本地连接补充用户名信息
                user_name_map = {}
                async with self._lock:
                    for uid, conns in self._local_connections.items():
                        if conns:
                            user_name_map[uid] = conns[0].username

                users = []
                for uid_str in raw:
                    try:
                        uid = int(uid_str)
                        users.append({
                            "user_id": uid,
                            "username": user_name_map.get(uid, ""),
                        })
                    except (ValueError, TypeError):
                        pass
                return users
            except Exception as e:
                logger.warning(f"获取在线用户失败，降级为本地查询: {e}")

        # 降级：仅本地查询
        async with self._lock:
            users = []
            seen = set()
            for user_id, connections in self._local_connections.items():
                if user_id not in seen and connections:
                    seen.add(user_id)
                    users.append(connections[0].to_dict())
            return users

    @property
    def online_count(self) -> int:
        """本地在线用户数"""
        return len(self._local_connections)

    # ===== 内部方法 =====

    async def _send_to_local_user(self, user_id: int, message: dict[str, Any]):
        """向本 Worker 的本地连接发送消息"""
        async with self._lock:
            connections = self._local_connections.get(user_id, [])[:]

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

        for ws in disconnected:
            await self._disconnect_local(ws, user_id)

    async def _broadcast_local(self, message: dict[str, Any], exclude_user: int | None = None):
        """向本 Worker 的所有本地连接广播"""
        async with self._lock:
            all_connections: list[WebSocketConnection] = []
            for user_id, conns in self._local_connections.items():
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

    async def _subscribe_loop(self):
        """订阅 Redis Pub/Sub 频道，接收跨 Worker 消息"""
        while self._started:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(settings.WS_REDIS_CHANNEL)
                logger.info(f"已订阅 WebSocket 频道: {settings.WS_REDIS_CHANNEL}")

                async for message in pubsub.listen():
                    if not self._started:
                        break
                    if message["type"] == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        await self._handle_pubsub_message(data)

                await pubsub.unsubscribe(settings.WS_REDIS_CHANNEL)
                await pubsub.aclose()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WebSocket Pub/Sub 异常: {e}, 3秒后重连")
                await asyncio.sleep(3)

    async def _handle_pubsub_message(self, data: str):
        """处理 Pub/Sub 消息"""
        try:
            msg = json.loads(data)
            target_user_id = msg.get("target_user_id")
            message_content = msg.get("message", {})
            exclude_user = msg.get("exclude_user")

            if target_user_id is not None:
                # 定向消息：只发给本 Worker 的本地连接
                await self._send_to_local_user(target_user_id, message_content)
            else:
                # 广播消息：发给所有本地连接
                await self._broadcast_local(message_content, exclude_user)
        except json.JSONDecodeError as e:
            logger.warning(f"Pub/Sub 消息解析失败: {e}")
        except Exception as e:
            logger.warning(f"处理 Pub/Sub 消息失败: {e}")

    async def _heartbeat_loop(self):
        """心跳循环：定期更新本 Worker 连接的用户的在线状态"""
        while self._started:
            try:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)

                async with self._lock:
                    user_ids = list(self._local_connections.keys())

                if user_ids and self._redis:
                    now = time.time()
                    pipe = self._redis.pipeline()
                    for uid in user_ids:
                        pipe.zadd(settings.WS_ONLINE_KEY, {str(uid): now})
                    # 清理过期用户
                    pipe.zremrangebyscore(
                        settings.WS_ONLINE_KEY, 0, now - settings.WS_ONLINE_TTL
                    )
                    await pipe.execute()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WebSocket 心跳失败: {e}")

    async def _update_online_status(self, user_id: int):
        """更新单个用户的在线状态"""
        if self._redis:
            try:
                await self._redis.zadd(
                    settings.WS_ONLINE_KEY,
                    {str(user_id): time.time()},
                )
            except Exception as e:
                logger.warning(f"更新 Redis 在线状态失败: user_id={user_id}, error={e}")


# 全局连接管理器
manager = DistributedConnectionManager()


class WebSocketService:
    """WebSocket 服务类（异步版本）"""

    @staticmethod
    async def verify_token(token: str) -> dict[str, Any] | None:
        """验证 JWT Token"""
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
        """广播服务器关闭通知（跨 Worker）"""
        try:
            await manager.broadcast({
                "type": "server_shutdown",
                "message": "服务器正在关闭，请保存工作",
                "reconnect": False,
            })
            logger.info("已广播关闭通知")
        except Exception as e:
            logger.error(f"广播关闭通知失败: {e}")

    @staticmethod
    async def handle_connection(websocket: WebSocket, token: str):
        """处理 WebSocket 连接"""
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
        """处理 WebSocket 消息"""
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


async def init_websocket_manager():
    """初始化 WebSocket 管理器（在 lifespan 中调用）"""
    redis = await get_redis_client()
    if redis:
        await manager.start(redis)
    else:
        logger.warning("Redis 不可用，WebSocket 以本地单 Worker 模式运行")


async def close_websocket_manager():
    """关闭 WebSocket 管理器（在 lifespan 中调用）"""
    await manager.stop()
