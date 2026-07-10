package mq

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

var ErrConsumerClosed = errors.New("consumer is closed")

// Handler 消息处理函数，接收带 TraceID 的 context
type Handler func(ctx context.Context, body []byte) error

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
func (c *Consumer) Consume(queue string, handler Handler) error {
	if c.closed.Load() {
		return ErrConsumerClosed
	}

	ch := c.getChannel()
	if ch == nil {
		return errors.New("rabbitmq channel not available")
	}

	// 声明队列（确保存在）
	_, err := ch.QueueDeclare(queue, true, false, false, false, nil)
	if err != nil {
		return err
	}

	// 绑定队列到交换机
	exchange := c.resolveExchange()
	routingKey := c.resolveRoutingKey(queue)
	if err := ch.QueueBind(queue, routingKey, exchange, false, nil); err != nil {
		return err
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

// handleMessage 处理单条消息，从 Headers 恢复 TraceID 和 traceparent
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

	// 调用业务处理函数
	if err := handler(ctx, d.Body); err != nil {
		c.logger.Warn("消息处理失败，重新入队",
			zap.String("trace_id", traceID),
			zap.Error(err))
		_ = d.Nack(false, true) // 重新入队
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
