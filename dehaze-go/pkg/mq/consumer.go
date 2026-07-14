package mq

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/database"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

var ErrConsumerClosed = errors.New("consumer is closed")

// MaxRetryCount MQ 最大重试次数，超过后消息进入死信队列
const MaxRetryCount = 3

const userIDHeader = "x-user-id"

// Handler 消息处理函数，接收带 TraceID 的 context
type Handler func(ctx context.Context, body []byte) error

// DLQHandler 死信队列处理函数
type DLQHandler func(ctx context.Context, body []byte, headers map[string]interface{}) error

// Consumer RabbitMQ 消费者
// 支持连接断开后自动重连（指数退避），对上层调用方透明
type Consumer struct {
	cfg    options.RabbitMQ
	logger *zap.Logger

	conn *amqp.Connection
	ch   *amqp.Channel
	mu   sync.RWMutex

	closed   atomic.Bool
	closeCh  chan struct{} // 通知重连协程和消费循环退出
	reconnWg sync.WaitGroup
}

// NewConsumer 创建 RabbitMQ 消费者
func NewConsumer(cfg options.RabbitMQ, logger *zap.Logger) *Consumer {
	if logger == nil {
		logger = zap.NewNop()
	}
	return &Consumer{
		cfg:     cfg,
		logger:  logger,
		closeCh: make(chan struct{}),
	}
}

// Connect 建立连接，同时启动连接断开监听
func (c *Consumer) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed.Load() {
		return ErrConsumerClosed
	}
	if c.ch != nil {
		return nil
	}
	return c.connectLocked()
}

// connectLocked 内部连接逻辑，调用方需持有写锁
func (c *Consumer) connectLocked() error {
	if c.cfg.URL == "" {
		return errors.New("rabbitmq url is empty")
	}

	conn, err := amqp.Dial(c.cfg.URL)
	if err != nil {
		return err
	}
	ch, err := conn.Channel()
	if err != nil {
		_ = conn.Close()
		return err
	}

	// 设置 QoS，限制未确认消息数量
	if err := ch.Qos(10, 0, false); err != nil {
		_ = ch.Close()
		_ = conn.Close()
		return err
	}

	c.conn = conn
	c.ch = ch

	// 启动断开监听，触发自动重连
	c.watchConnection(conn)
	return nil
}

// watchConnection 监听连接关闭事件，触发重连
func (c *Consumer) watchConnection(conn *amqp.Connection) {
	connCloseCh := conn.NotifyClose(make(chan *amqp.Error, 1))

	c.reconnWg.Add(1)
	go func() {
		defer c.reconnWg.Done()
		select {
		case amqpErr, ok := <-connCloseCh:
			if !ok || c.closed.Load() {
				return
			}
			c.logger.Warn("RabbitMQ Consumer 连接断开，开始自动重连", zap.Error(amqpErr))
			c.handleDisconnect()
			c.reconnectLoop()
		case <-c.closeCh:
			return
		}
	}()
}

// handleDisconnect 清理已断开的连接资源
func (c *Consumer) handleDisconnect() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.ch != nil {
		_ = c.ch.Close()
		c.ch = nil
	}
	if c.conn != nil {
		_ = c.conn.Close()
		c.conn = nil
	}
}

// reconnectLoop 指数退避重连循环
func (c *Consumer) reconnectLoop() {
	cfg := c.reconnectConfig()
	attempt := 0

	for {
		if c.closed.Load() {
			return
		}
		if cfg.maxRetries > 0 && attempt >= cfg.maxRetries {
			c.logger.Error("RabbitMQ Consumer 重连已达最大重试次数，放弃重连",
				zap.Int("maxRetries", cfg.maxRetries))
			return
		}

		interval := c.backoffInterval(attempt, cfg.initialInterval, cfg.maxInterval)
		attempt++

		c.logger.Info("RabbitMQ Consumer 重连等待中",
			zap.Int("attempt", attempt),
			zap.Duration("interval", interval))

		select {
		case <-time.After(interval):
		case <-c.closeCh:
			return
		}

		c.mu.Lock()
		err := c.connectLocked()
		c.mu.Unlock()

		if err == nil {
			c.logger.Info("RabbitMQ Consumer 重连成功", zap.Int("attempt", attempt))
			return
		}
		c.logger.Warn("RabbitMQ Consumer 重连失败",
			zap.Int("attempt", attempt),
			zap.Error(err))
	}
}

type consumerReconnectCfg struct {
	maxRetries      int
	initialInterval time.Duration
	maxInterval     time.Duration
}

func (c *Consumer) reconnectConfig() consumerReconnectCfg {
	cfg := consumerReconnectCfg{
		maxRetries:      c.cfg.ReconnectMaxRetries,
		initialInterval: c.cfg.ReconnectInitialInterval,
		maxInterval:     c.cfg.ReconnectMaxInterval,
	}
	if cfg.initialInterval <= 0 {
		cfg.initialInterval = 1 * time.Second
	}
	if cfg.maxInterval <= 0 {
		cfg.maxInterval = 30 * time.Second
	}
	return cfg
}

// backoffInterval 计算指数退避间隔: min(initial * 2^attempt, max)
func (c *Consumer) backoffInterval(attempt int, initial, max time.Duration) time.Duration {
	backoff := time.Duration(float64(initial) * math.Pow(2, float64(attempt)))
	if backoff > max {
		backoff = max
	}
	return backoff
}

// Consume 开始消费指定队列的消息
// queue: 队列名称
// handler: 消息处理函数，接收带 TraceID 的 context
// 自动声明 DLX 交换机和死信队列，消息处理失败超过 MaxRetryCount 后进入死信队列
func (c *Consumer) Consume(queue string, handler Handler) error {
	if c.closed.Load() {
		return ErrConsumerClosed
	}

	ch := c.getChannel()
	if ch == nil {
		return errors.New("rabbitmq channel not available")
	}

	exchange := c.resolveExchange()
	dlxExchange := c.resolveDlxExchange()
	dlqName := queue + ".dlq"
	dlqRoutingKey := c.resolveDlxRoutingKey(queue)

	// 声明 DLX 交换机（direct 类型）
	if err := ch.ExchangeDeclare(dlxExchange, "direct", true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare DLX exchange: %w", err)
	}

	// 声明主队列（带 DLX 参数，消息过期/reject 后进入死信交换机）
	queueArgs := amqp.Table{
		"x-dead-letter-exchange":    dlxExchange,
		"x-dead-letter-routing-key": dlqRoutingKey,
	}
	if _, err := ch.QueueDeclare(queue, true, false, false, false, queueArgs); err != nil {
		return err
	}

	// 绑定主队列到交换机
	routingKey := c.resolveRoutingKey(queue)
	if err := ch.QueueBind(queue, routingKey, exchange, false, nil); err != nil {
		return err
	}

	// 声明死信队列
	if _, err := ch.QueueDeclare(dlqName, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare DLQ: %w", err)
	}

	// 绑定死信队列到 DLX 交换机
	if err := ch.QueueBind(dlqName, dlqRoutingKey, dlxExchange, false, nil); err != nil {
		return fmt.Errorf("bind DLQ: %w", err)
	}

	deliveries, err := ch.Consume(queue, "", false, false, false, false, nil)
	if err != nil {
		return err
	}

	c.reconnWg.Add(1)
	go func() {
		defer c.reconnWg.Done()
		for {
			select {
			case <-c.closeCh:
				return
			case d, ok := <-deliveries:
				if !ok {
					return
				}
				c.handleMessage(d, handler)
			}
		}
	}()

	return nil
}

// ConsumeDLQ 开始消费死信队列的消息
func (c *Consumer) ConsumeDLQ(queue string, handler DLQHandler) error {
	if c.closed.Load() {
		return ErrConsumerClosed
	}

	ch := c.getChannel()
	if ch == nil {
		return errors.New("rabbitmq channel not available")
	}

	dlqName := queue + ".dlq"

	// 确保死信队列存在
	if _, err := ch.QueueDeclare(dlqName, true, false, false, false, nil); err != nil {
		return err
	}

	deliveries, err := ch.Consume(dlqName, "", false, false, false, false, nil)
	if err != nil {
		return err
	}

	c.reconnWg.Add(1)
	go func() {
		defer c.reconnWg.Done()
		for {
			select {
			case <-c.closeCh:
				return
			case d, ok := <-deliveries:
				if !ok {
					return
				}
				// 从 AMQP Headers 恢复 TraceID
				traceID, traceParent := c.extractTraceInfo(d.Headers)
				if traceID == "" {
					traceID = trace.NewTraceID()
				}
				if traceParent == "" {
					traceParent = trace.NewTraceParent(traceID)
				}
				ctx := trace.WithTraceID(context.Background(), traceID)
				ctx = trace.WithTraceParent(ctx, traceParent)
				ctx = trace.WithLogger(ctx, c.logger.With(zap.String(trace.TraceFieldName, traceID)))

				// 注入 userID：死信处理为系统行为，从 Headers 恢复原始用户身份，缺失时兜底为 SystemUserID
				userID := common.SystemUserID
				if uid, ok := d.Headers[userIDHeader]; ok {
					switch n := uid.(type) {
					case int64:
						userID = n
					case int32:
						userID = int64(n)
					case int:
						userID = int64(n)
					case float64:
						userID = int64(n)
					}
				}
				ctx = database.SetUserID(ctx, userID)

				if err := handler(ctx, d.Body, map[string]interface{}(d.Headers)); err != nil {
					c.logger.Warn("死信队列消息处理失败", zap.String("trace_id", traceID), zap.Error(err))
				}
				_ = d.Ack(false)
			}
		}
	}()

	return nil
}

// handleMessage 处理单条消息，从 Headers 恢复 TraceID 和 traceparent
// 处理失败时按 x-retry-count 进行分级重试，超阈值后 reject 入死信队列
func (c *Consumer) handleMessage(d amqp.Delivery, handler Handler) {
	// 从 AMQP Headers 恢复 TraceID 和 traceparent
	traceID, traceParent := c.extractTraceInfo(d.Headers)
	if traceID == "" {
		traceID = trace.NewTraceID()
	}
	if traceParent == "" {
		// 没有 traceparent 则生成新的
		traceParent = trace.NewTraceParent(traceID)
	}

	// 构建 context 并注入 TraceID 和 traceparent
	ctx := trace.WithTraceID(context.Background(), traceID)
	ctx = trace.WithTraceParent(ctx, traceParent)

	// 缓存带 TraceID 的 logger，供整条链路复用
	ctx = trace.WithLogger(ctx, c.logger.With(zap.String(trace.TraceFieldName, traceID)))

	userID := common.SystemUserID
	if uid, ok := d.Headers[userIDHeader]; ok {
		switch n := uid.(type) {
		case int64:
			userID = n
		case int32:
			userID = int64(n)
		case int:
			userID = int64(n)
		case float64:
			userID = int64(n)
		}
	}
	ctx = database.SetUserID(ctx, userID)

	// 调用业务处理函数
	if err := handler(ctx, d.Body); err != nil {
		retryCount := c.getRetryCount(d.Headers)
		if retryCount < MaxRetryCount {
			// 未超阈值：递增 retry-count 后重新发布到主队列
			newRetryCount := retryCount + 1
			c.logger.Warn("消息处理失败，重新入队重试",
				zap.String("trace_id", traceID),
				zap.Int("retryCount", newRetryCount),
				zap.Int("maxRetry", MaxRetryCount),
				zap.Error(err))
			c.republishWithRetryCount(d, newRetryCount)
		} else {
			// 超阈值：reject 入死信队列（Nack with requeue=false → DLX → DLQ）
			c.logger.Error("消息处理失败且已达最大重试次数，进入死信队列",
				zap.String("trace_id", traceID),
				zap.Int("retryCount", retryCount),
				zap.Error(err))
			_ = d.Nack(false, false)
		}
		return
	}

	_ = d.Ack(false)
}

// getRetryCount 从 AMQP Headers 获取 x-retry-count
func (c *Consumer) getRetryCount(headers amqp.Table) int {
	if headers == nil {
		return 0
	}
	if v, ok := headers["x-retry-count"]; ok {
		switch n := v.(type) {
		case int:
			return n
		case int32:
			return int(n)
		case int64:
			return int(n)
		}
	}
	return 0
}

// republishWithRetryCount 递增 x-retry-count 后重新发布消息到主队列
func (c *Consumer) republishWithRetryCount(d amqp.Delivery, retryCount int) {
	ch := c.getChannel()
	if ch == nil {
		// channel 不可用时直接 reject 入死信
		_ = d.Nack(false, false)
		return
	}

	headers := d.Headers
	if headers == nil {
		headers = amqp.Table{}
	}
	headers["x-retry-count"] = retryCount

	exchange := c.resolveExchange()
	routingKey := d.RoutingKey

	err := ch.PublishWithContext(context.Background(), exchange, routingKey, false, false, amqp.Publishing{
		ContentType:  d.ContentType,
		Body:         d.Body,
		DeliveryMode: amqp.Persistent,
		Headers:      headers,
	})
	if err != nil {
		c.logger.Error("重新发布消息失败，直接 reject 入死信", zap.Error(err))
		_ = d.Nack(false, false)
		return
	}
	_ = d.Ack(false)
}

// extractTraceInfo 从 AMQP Headers 中提取 TraceID 和 traceparent
func (c *Consumer) extractTraceInfo(headers amqp.Table) (string, string) {
	if headers == nil {
		return "", ""
	}

	var traceID, traceParent string

	// 尝试从 Headers 获取 X-Trace-ID
	if v, ok := headers[trace.HeaderName]; ok {
		if s, ok := v.(string); ok {
			traceID = trace.NormalizeTraceID(s)
		}
	}

	// 尝试从 Headers 获取 traceparent
	if v, ok := headers[trace.HeaderNameTraceParent]; ok {
		if s, ok := v.(string); ok {
			traceParent = strings.TrimSpace(s)
			// 校验 traceparent 格式，非法则丢弃
			if trace.ParseTraceParent(traceParent) == "" {
				traceParent = ""
			}
		}
	}

	return traceID, traceParent
}

func (c *Consumer) getChannel() *amqp.Channel {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.ch
}

// IsConnected 返回当前 RabbitMQ 连接是否活跃
func (c *Consumer) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conn != nil && !c.conn.IsClosed()
}

// Close 关闭连接并停止消费
func (c *Consumer) Close() error {
	if !c.closed.CompareAndSwap(false, true) {
		return nil
	}
	close(c.closeCh)

	c.reconnWg.Wait()

	c.mu.Lock()
	defer c.mu.Unlock()

	var err error
	if c.ch != nil {
		err = c.ch.Close()
		c.ch = nil
	}
	if c.conn != nil {
		closeErr := c.conn.Close()
		if err == nil {
			err = closeErr
		}
		c.conn = nil
	}
	return err
}

func (c *Consumer) resolveExchange() string {
	if c.cfg.Exchange != "" {
		return c.cfg.Exchange
	}
	return "dehaze.tasks"
}

func (c *Consumer) resolveRoutingKey(queue string) string {
	if c.cfg.RoutingKeyPrefix != "" {
		return c.cfg.RoutingKeyPrefix + "." + queue
	}
	return "task." + queue
}

// resolveDlxExchange 返回死信交换机名称（主交换机 + .dlx 后缀）
func (c *Consumer) resolveDlxExchange() string {
	return c.resolveExchange() + ".dlx"
}

// resolveDlxRoutingKey 返回死信路由键（主路由键 + .dlx 后缀）
func (c *Consumer) resolveDlxRoutingKey(queue string) string {
	return c.resolveRoutingKey(queue) + ".dlx"
}
