package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// CacheInvalidationMsg 缓存失效消息
type CacheInvalidationMsg struct {
	Type     string `json:"type"`     // 消息类型: "permission", "role", "user" 等
	Key      string `json:"key"`      // 缓存键
	SenderID string `json:"senderId"` // 发送者标识，用于忽略自己发送的消息
}

// PubSub Redis 发布订阅封装
type PubSub struct {
	client   *redis.Client
	channel  string
	senderID string // 实例标识，用于区分消息来源

	mu          sync.RWMutex
	subscribers map[string][]func(CacheInvalidationMsg) // 按消息类型分组的订阅者
	stopChan    chan struct{}
	stopped     bool

	// 并发控制
	semaphore chan struct{} // 用于限制handler并发数
}

// pubsubInstance 单例实例
var (
	pubsubInstance *PubSub
	pubsubOnce     sync.Once
)

// GetPubSub 获取 PubSub 单例
func GetPubSub() *PubSub {
	return pubsubInstance
}

// InitPubSub 初始化 PubSub
// channel: 订阅的频道名称
// senderID: 实例标识，建议使用 hostname 或 pod name
// maxConcurrency: handler最大并发数，建议16-32
func InitPubSub(channel, senderID string, maxConcurrency int) (*PubSub, error) {
	if _client == nil {
		return nil, fmt.Errorf("redis client not initialized")
	}

	if maxConcurrency <= 0 {
		maxConcurrency = 16 // 默认并发数
	}

	var initErr error
	pubsubOnce.Do(func() {
		pubsubInstance = &PubSub{
			client:      _client,
			channel:     channel,
			senderID:    senderID,
			subscribers: make(map[string][]func(CacheInvalidationMsg)),
			stopChan:    make(chan struct{}),
			semaphore:   make(chan struct{}, maxConcurrency),
		}

		// 启动订阅监听
		if err := pubsubInstance.start(); err != nil {
			initErr = err
			pubsubInstance = nil
		}
	})

	return pubsubInstance, initErr
}

// start 启动订阅监听
func (ps *PubSub) start() error {
	ctx := context.Background()
	pubsub := ps.client.Subscribe(ctx, ps.channel)

	// 验证订阅是否成功
	_, err := pubsub.Receive(ctx)
	if err != nil {
		logger.Error("Redis Pub/Sub 订阅失败", zap.String("channel", ps.channel), zap.Error(err))
		return fmt.Errorf("subscribe failed: %w", err)
	}

	logger.Info("Redis Pub/Sub 订阅成功", zap.String("channel", ps.channel), zap.String("senderId", ps.senderID))

	// 启动消息处理协程
	go func() {
		ch := pubsub.Channel()
		for {
			select {
			case <-ps.stopChan:
				pubsub.Close()
				logger.Info("Redis Pub/Sub 订阅已停止")
				return
			case msg, ok := <-ch:
				if !ok {
					return
				}
				ps.handleMessage(msg)
			}
		}
	}()

	return nil
}

// handleMessage 处理接收到的消息
func (ps *PubSub) handleMessage(msg *redis.Message) {
	var cacheMsg CacheInvalidationMsg
	if err := json.Unmarshal([]byte(msg.Payload), &cacheMsg); err != nil {
		logger.Warn("解析缓存失效消息失败", zap.String("payload", msg.Payload), zap.Error(err))
		return
	}

	// 忽略自己发送的消息
	if cacheMsg.SenderID == ps.senderID {
		return
	}

	logger.Debug("收到缓存失效消息",
		zap.String("type", cacheMsg.Type),
		zap.String("key", cacheMsg.Key),
		zap.String("from", cacheMsg.SenderID),
	)

	// 通知订阅者（使用semaphore控制并发）
	ps.mu.RLock()
	handlers, ok := ps.subscribers[cacheMsg.Type]
	ps.mu.RUnlock()

	if ok {
		for _, handler := range handlers {
			// 获取信号量，控制并发数
			ps.semaphore <- struct{}{}
			go func(h func(CacheInvalidationMsg)) {
				defer func() {
					<-ps.semaphore // 释放信号量
					if r := recover(); r != nil {
						logger.Error("缓存失效消息处理异常", zap.Any("panic", r))
					}
				}()
				h(cacheMsg)
			}(handler)
		}
	}
}

// Publish 发布缓存失效消息
func (ps *PubSub) Publish(ctx context.Context, msgType, key string) error {
	msg := CacheInvalidationMsg{
		Type:     msgType,
		Key:      key,
		SenderID: ps.senderID,
	}

	payload, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal message failed: %w", err)
	}

	if err := ps.client.Publish(ctx, ps.channel, payload).Err(); err != nil {
		logger.Error("发布缓存失效消息失败",
			zap.String("type", msgType),
			zap.String("key", key),
			zap.Error(err),
		)
		return err
	}

	logger.Debug("发布缓存失效消息成功",
		zap.String("type", msgType),
		zap.String("key", key),
	)
	return nil
}

// Subscribe 订阅指定类型的缓存失效消息
func (ps *PubSub) Subscribe(msgType string, handler func(CacheInvalidationMsg)) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	ps.subscribers[msgType] = append(ps.subscribers[msgType], handler)
	logger.Debug("订阅缓存失效消息", zap.String("type", msgType))
}

// Stop 停止订阅
func (ps *PubSub) Stop() {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	if ps.stopped {
		return
	}

	ps.stopped = true
	close(ps.stopChan)
	logger.Info("Redis Pub/Sub 已停止")
}

// IsStopped 检查是否已停止
func (ps *PubSub) IsStopped() bool {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return ps.stopped
}
