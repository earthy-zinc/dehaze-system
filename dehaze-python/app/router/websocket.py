from app.service.websocket_service import WebSocketService
from fastapi import APIRouter, Query, WebSocket

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    sessionId: str = Query(..., description="Session ID 用于认证"),
):
    """
    WebSocket 连接端点

    连接方式: ws://host/ws?sessionId=YOUR_SESSION_ID

    支持的消息类型:
    - ping: 心跳检测，返回 pong
    - broadcast: 广播消息给所有在线用户
    - private: 发送私信给指定用户
    - get_online_users: 获取在线用户列表

    服务器推送的消息类型:
    - connected: 连接成功
    - broadcast: 广播消息
    - private_message: 私信消息
    - user_online: 用户上线通知
    - user_offline: 用户下线通知
    - online_users: 在线用户列表
    - error: 错误消息
    """
    await WebSocketService.handle_connection(websocket, sessionId)
