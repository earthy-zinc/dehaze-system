package websocket

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	// 允许所有来源（与 Java/Python 对齐，CORS 由网关统一处理）
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

const (
	writeWait      = 10 * time.Second
	pongWait       = 60 * time.Second
	pingPeriod     = 30 * time.Second
	maxMessageSize = 4096
)

// HandleWebSocket 处理 WebSocket 连接请求
// 通过 query 参数 token 进行 JWT 认证（对齐 Python 端方案）
func HandleWebSocket(c *gin.Context) {
	tokenStr := c.Query("token")
	if tokenStr == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"message": "认证失败，请重新登录"})
		return
	}

	jwtUtil := security.NewJWT()
	claims, err := jwtUtil.ParseToken(tokenStr)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"message": "认证失败，请重新登录"})
		return
	}

	userID := claims.UserID

	manager := GetManager()
	if manager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"message": "WebSocket 服务未就绪"})
		return
	}

	// 升级为 WebSocket 连接
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.Error("WebSocket 升级失败", zap.Error(err))
		return
	}

	// 注册连接
	wsConn := manager.Register(userID, conn)
	defer func() {
		manager.Unregister(wsConn)
		_ = conn.Close()
	}()

	// 发送连接成功消息
	welcome, _ := json.Marshal(map[string]interface{}{
		"type":    "connected",
		"message": "WebSocket 连接成功",
		"user_id": userID,
	})
	_ = conn.WriteMessage(websocket.TextMessage, welcome)

	// 启动 ping 计时器
	go func() {
		ticker := time.NewTicker(pingPeriod)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				conn.SetWriteDeadline(time.Now().Add(writeWait))
				if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
					return
				}
			}
		}
	}()

	// 读取循环
	conn.SetReadLimit(maxMessageSize)
	conn.SetReadDeadline(time.Now().Add(pongWait))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, message, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				logger.Debug("WebSocket 读取异常", zap.Int64("userID", userID), zap.Error(err))
			}
			break
		}

		// 处理客户端消息（ping 心跳）
		var msg map[string]interface{}
		if err := json.Unmarshal(message, &msg); err != nil {
			continue
		}
		if msg["type"] == "ping" {
			pong, _ := json.Marshal(map[string]interface{}{"type": "pong"})
			conn.SetWriteDeadline(time.Now().Add(writeWait))
			_ = conn.WriteMessage(websocket.TextMessage, pong)
		}
	}
}
