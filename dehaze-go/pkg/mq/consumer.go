package mq

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
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

	// topologies 记录已声明的队列拓扑，重连后用于重新声明队列与交换机
	topologies []queueTopology
}

// queueTopology 描述一次消费订阅涉及的队列/交换机拓扑，用于重连后重新声明
type queueTopology struct {
	queue         string
	exchange      string
	dlxExchange   string
	dlqName       string
	routingKey    string
	dlqRoutingKey string
	queueArgs     amqp.Table
	retryQueues   []retryQueueSpec
	// dlqOnly 为 true 时仅声明死信队列（ConsumeDLQ 场景）
	dlqOnly bool
}

type retryQueueSpec struct {
	name       string
	routingKey string
	args       amqp.Table
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
			// 重连成功后重新声明所有已记录的队列与交换机拓扑，避免队列丢失导致消息无法路由
			if rerr := c.redeclareAll(); rerr != nil {
				c.logger.Error("重连后重新声明队列拓扑失败", zap.Error(rerr))
			}
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
	dlqName := queue + ".dlx"
	routingKey := c.resolveRoutingKey(queue)
	dlqRoutingKey := c.resolveDlxRoutingKey(queue)

	queueArgs := amqp.Table{
		"x-message-ttl":             int32(86400000),
		"x-dead-letter-exchange":    dlxExchange,
		"x-dead-letter-routing-key": dlqRoutingKey,
	}

	topo := queueTopology{
		queue:         queue,
		exchange:      exchange,
		dlxExchange:   dlxExchange,
		dlqName:       dlqName,
		routingKey:    routingKey,
		dlqRoutingKey: dlqRoutingKey,
		queueArgs:     queueArgs,
		retryQueues:   c.buildRetryQueues(queue, exchange, routingKey),
	}
	if err := c.declareTopology(ch, topo); err != nil {
		return err
	}
	c.recordTopology(topo)

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

	dlqName := queue + ".dlx"

	// 确保死信队列存在
	topo := queueTopology{
		queue:   queue,
		dlqName: dlqName,
		dlqOnly: true,
	}
	if err := c.declareTopology(ch, topo); err != nil {
		return err
	}
	c.recordTopology(topo)

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
				userID := extractUserID(d.Headers)
				ctx = database.SetUserID(ctx, userID)

				if err := handler(ctx, d.Body, map[string]interface{}(d.Headers)); err != nil {
					c.logger.Warn("死信队列消息处理失败，Nack 拒绝以防数据丢失", zap.String("trace_id", traceID), zap.Error(err))
					// 处理失败时不 Ack，requeue=false 以便进入二级死信或人工介入，避免丢消息
					_ = d.Nack(false, false)
					continue
				}
				_ = d.Ack(false)
			}
		}
	}()

	return nil
}

// handleMessage 处理单条消息，从 Headers 恢复 TraceID 和 traceparent
// 处理失败时按 x-retry-count 投递到对应级别的 TTL 重试队列，超阈值后 reject 入死信队列
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

	userID := extractUserID(d.Headers)
	ctx = database.SetUserID(ctx, userID)

	// 调用业务处理函数
	if err := handler(ctx, d.Body); err != nil {
		retryCount := c.getRetryCount(d.Headers)
		if retryCount < MaxRetryCount {
			newRetryCount := retryCount + 1
			c.logger.Warn("消息处理失败，投递到重试队列",
				zap.String("trace_id", traceID),
				zap.Int("retryCount", newRetryCount),
				zap.Int("maxRetry", MaxRetryCount),
				zap.Error(err))
			c.publishToRetryQueue(d, newRetryCount)
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

// extractUserID 从 AMQP Headers 提取用户ID，缺失时返回 SystemUserID
// AMQP message headers 中的数值经 JSON/AMQP 编码后可能为多种数值类型，统一兜底转换
func extractUserID(headers amqp.Table) int64 {
	if headers == nil {
		return common.SystemUserID
	}
	uid, ok := headers[userIDHeader]
	if !ok {
		return common.SystemUserID
	}
	switch n := uid.(type) {
	case int64:
		return n
	case int32:
		return int64(n)
	case int:
		return int64(n)
	case float64:
		return int64(n)
	default:
		return common.SystemUserID
	}
}

// publishToRetryQueue 将失败消息投递到对应级别的 TTL 重试队列
func (c *Consumer) publishToRetryQueue(d amqp.Delivery, retryCount int) {
	ch := c.getChannel()
	if ch == nil {
		_ = d.Nack(false, false)
		return
	}

	headers := d.Headers
	if headers == nil {
		headers = amqp.Table{}
	}
	headers["x-retry-count"] = retryCount

	queue := d.RoutingKey

	retryQueue := queue + ".retry." + strconv.Itoa(retryCount-1)
	routingKey := c.resolveRoutingKey(retryQueue)
	exchange := c.resolveExchange()

	err := ch.PublishWithContext(context.Background(), exchange, routingKey, false, false, amqp.Publishing{
		ContentType:  d.ContentType,
		Body:         d.Body,
		DeliveryMode: amqp.Persistent,
		Headers:      headers,
	})
	if err != nil {
		c.logger.Error("投递消息到重试队列失败，直接 reject 入死信", zap.Error(err))
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
	return queue
}

// resolveDlxExchange 返回死信交换机名称（主交换机 + .dlx 后缀）
func (c *Consumer) resolveDlxExchange() string {
	return c.resolveExchange() + ".dlx"
}

// resolveDlxRoutingKey 返回死信路由键（主路由键 + .dlx 后缀）
func (c *Consumer) resolveDlxRoutingKey(queue string) string {
	return c.resolveRoutingKey(queue + ".dlx")
}

// declareTopology 在指定 channel 上声明队列拓扑（交换机、主队列、死信队列及绑定）
// dlqOnly 为 true 时仅声明死信队列（ConsumeDLQ 场景）
func (c *Consumer) declareTopology(ch *amqp.Channel, topo queueTopology) error {
	if topo.dlqOnly {
		if _, err := ch.QueueDeclare(topo.dlqName, true, false, false, false, nil); err != nil {
			return fmt.Errorf("declare DLQ: %w", err)
		}
		return nil
	}

	// 声明 DLX 交换机（direct 类型）
	if err := ch.ExchangeDeclare(topo.dlxExchange, "direct", true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare DLX exchange: %w", err)
	}

	// 声明主队列（带 DLX 参数，消息过期/reject 后进入死信交换机）
	if _, err := ch.QueueDeclare(topo.queue, true, false, false, false, topo.queueArgs); err != nil {
		return fmt.Errorf("declare queue: %w", err)
	}

	// 绑定主队列到交换机
	if err := ch.QueueBind(topo.queue, topo.routingKey, topo.exchange, false, nil); err != nil {
		return fmt.Errorf("bind queue: %w", err)
	}

	// 声明死信队列
	if _, err := ch.QueueDeclare(topo.dlqName, true, false, false, false, nil); err != nil {
		return fmt.Errorf("declare DLQ: %w", err)
	}

	// 绑定死信队列到 DLX 交换机
	if err := ch.QueueBind(topo.dlqName, topo.dlqRoutingKey, topo.dlxExchange, false, nil); err != nil {
		return fmt.Errorf("bind DLQ: %w", err)
	}

	for _, rq := range topo.retryQueues {
		if _, err := ch.QueueDeclare(rq.name, true, false, false, false, rq.args); err != nil {
			return fmt.Errorf("declare retry queue %s: %w", rq.name, err)
		}
		if err := ch.QueueBind(rq.name, rq.routingKey, topo.exchange, false, nil); err != nil {
			return fmt.Errorf("bind retry queue %s: %w", rq.name, err)
		}
	}
	return nil
}

var retryDelays = []int32{5000, 30000, 300000}

func (c *Consumer) buildRetryQueues(queue, exchange, mainRoutingKey string) []retryQueueSpec {
	queues := make([]retryQueueSpec, 0, MaxRetryCount)
	for i := 0; i < MaxRetryCount; i++ {
		name := queue + ".retry." + strconv.Itoa(i)
		queues = append(queues, retryQueueSpec{
			name:       name,
			routingKey: c.resolveRoutingKey(name),
			args: amqp.Table{
				"x-message-ttl":             retryDelays[i],
				"x-dead-letter-exchange":    exchange,
				"x-dead-letter-routing-key": mainRoutingKey,
			},
		})
	}
	return queues
}

// recordTopology 记录已声明的拓扑，用于重连后重新声明。重复声明是幂等的，无需去重
func (c *Consumer) recordTopology(topo queueTopology) {
	c.mu.Lock()
	c.topologies = append(c.topologies, topo)
	c.mu.Unlock()
}

// redeclareAll 重连成功后在当前 channel 上重新声明所有已记录的拓扑
func (c *Consumer) redeclareAll() error {
	c.mu.RLock()
	topologies := make([]queueTopology, len(c.topologies))
	copy(topologies, c.topologies)
	c.mu.RUnlock()

	ch := c.getChannel()
	if ch == nil {
		return errors.New("rabbitmq channel not available")
	}

	for _, topo := range topologies {
		if err := c.declareTopology(ch, topo); err != nil {
			return fmt.Errorf("redeclare topology for queue %s: %w", topo.queue, err)
		}
	}
	if len(topologies) > 0 {
		c.logger.Info("重连后已重新声明队列拓扑", zap.Int("count", len(topologies)))
	}
	return nil
}
