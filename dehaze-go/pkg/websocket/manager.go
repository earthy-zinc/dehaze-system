package websocket

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gorilla/websocket"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// WS_CHANNEL WebSocket 跨实例广播频道（对齐 Python/Java）
const WS_CHANNEL = "dehaze:ws:broadcast"

// wsEnvelope Redis Pub/Sub 消息信封
type wsEnvelope struct {
	TargetUserID int64           `json:"target_user_id"`
	Message      json.RawMessage `json:"message"`
}

// Connection WebSocket 连接信息
type Connection struct {
	conn   *websocket.Conn
	userID int64
}

// Manager WebSocket 连接管理器
// 通过 Redis Pub/Sub 实现跨实例消息投递，对齐 Python 端方案。
type Manager struct {
	mu       sync.RWMutex
	connections map[int64][]*Connection // 本地连接（一个用户可能有多连接）

	redisClient *redis.Client
	stopChan    chan struct{}
	stopped     bool
}

var (
	instance *Manager
	once     sync.Once
)

// GetManager 获取 WebSocket 管理器单例
func GetManager() *Manager {
	return instance
}

// InitManager 初始化 WebSocket 管理器并订阅 Redis Pub/Sub
func InitManager(redisClient *redis.Client) (*Manager, error) {
	var initErr error
	once.Do(func() {
		instance = &Manager{
			connections: make(map[int64][]*Connection),
			redisClient: redisClient,
			stopChan:    make(chan struct{}),
		}
		if err := instance.subscribe(); err != nil {
			initErr = err
			instance = nil
		}
	})
	return instance, initErr
}

// subscribe 订阅 Redis Pub/Sub 频道
func (m *Manager) subscribe() error {
	ctx := context.Background()
	pubsub := m.redisClient.Subscribe(ctx, WS_CHANNEL)

	// 验证订阅成功
	if _, err := pubsub.Receive(ctx); err != nil {
		logger.Error("WebSocket Pub/Sub 订阅失败", zap.String("channel", WS_CHANNEL), zap.Error(err))
		return fmt.Errorf("websocket pubsub subscribe failed: %w", err)
	}

	logger.Info("WebSocket Pub/Sub 订阅成功", zap.String("channel", WS_CHANNEL))

	go func() {
		ch := pubsub.Channel()
		for {
			select {
			case <-m.stopChan:
				pubsub.Close()
				logger.Info("WebSocket Pub/Sub 订阅已停止")
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				m.handlePubSubMessage(msg)
			}
		}
	}()

	return nil
}

// handlePubSubMessage 处理 Redis Pub/Sub 消息，投递给本地连接
func (m *Manager) handlePubSubMessage(msg *redis.Message) {
	var envelope wsEnvelope
	if err := json.Unmarshal([]byte(msg.Payload), &envelope); err != nil {
		logger.Warn("解析 WebSocket Pub/Sub 消息失败", zap.String("payload", msg.Payload), zap.Error(err))
		return
	}

	m.sendToLocalUser(envelope.TargetUserID, envelope.Message)
}

// sendToLocalUser 向本实例上的用户连接发送消息
func (m *Manager) sendToLocalUser(userID int64, data []byte) {
	m.mu.RLock()
	conns := make([]*Connection, len(m.connections[userID]))
	copy(conns, m.connections[userID])
	m.mu.RUnlock()

	for _, c := range conns {
		_ = c.conn.WriteMessage(websocket.TextMessage, data)
	}
}

// Register 注册一个 WebSocket 连接
func (m *Manager) Register(userID int64, conn *websocket.Conn) *Connection {
	c := &Connection{conn: conn, userID: userID}

	m.mu.Lock()
	m.connections[userID] = append(m.connections[userID], c)
	m.mu.Unlock()

	logger.Debug("WebSocket 连接已注册", zap.Int64("userID", userID))
	return c
}

// Unregister 注销一个 WebSocket 连接
func (m *Manager) Unregister(c *Connection) {
	m.mu.Lock()
	defer m.mu.Unlock()

	conns := m.connections[c.userID]
	for i, conn := range conns {
		if conn == c {
			m.connections[c.userID] = append(conns[:i], conns[i+1:]...)
			break
		}
	}
	if len(m.connections[c.userID]) == 0 {
		delete(m.connections, c.userID)
	}

	logger.Debug("WebSocket 连接已注销", zap.Int64("userID", c.userID))
}

// PublishToUser 发布消息到指定用户（跨实例）
func (m *Manager) PublishToUser(userID int64, message map[string]interface{}) {
	msgBytes, err := json.Marshal(message)
	if err != nil {
		logger.Debug("WebSocket 消息序列化失败", zap.Error(err))
		return
	}

	envelope := wsEnvelope{
		TargetUserID: userID,
		Message:      msgBytes,
	}
	payload, err := json.Marshal(envelope)
	if err != nil {
		logger.Debug("WebSocket 信封序列化失败", zap.Error(err))
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := m.redisClient.Publish(ctx, WS_CHANNEL, payload).Err(); err != nil {
		logger.Debug("Redis Pub/Sub 发布失败", zap.Error(err))
	}
}

// Stop 停止管理器
func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.stopped {
		return
	}
	m.stopped = true
	close(m.stopChan)

	// 关闭所有本地连接
	for userID, conns := range m.connections {
		for _, c := range conns {
			_ = c.conn.WriteMessage(websocket.CloseMessage,
				websocket.FormatCloseMessage(websocket.CloseNormalClosure, "服务器关闭"))
			_ = c.conn.Close()
		}
		delete(m.connections, userID)
	}

	logger.Info("WebSocket 管理器已停止")
}
