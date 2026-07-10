package mq

import (
	"context"
	"errors"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

var ErrPublisherClosed = errors.New("publisher is closed")

// Publisher RabbitMQ 发布器
// 支持连接断开后自动重连（指数退避），对上层调用方透明
type Publisher struct {
	cfg    options.RabbitMQ
	logger *zap.Logger

	conn *amqp.Connection
	ch   *amqp.Channel
	mu   sync.RWMutex

	closed   atomic.Bool
	closeCh  chan struct{} // 通知重连协程退出
	reconnWg sync.WaitGroup
}

// NewPublisher 创建 RabbitMQ 发布器
func NewPublisher(cfg options.RabbitMQ, logger *zap.Logger) *Publisher {
	if logger == nil {
		logger = zap.NewNop()
	}
	return &Publisher{
		cfg:     cfg,
		logger:  logger,
		closeCh: make(chan struct{}),
	}
}

// Connect 建立连接并声明交换机，同时启动连接断开监听
func (p *Publisher) Connect() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed.Load() {
		return ErrPublisherClosed
	}
	if p.ch != nil {
		return nil
	}
	return p.connectLocked()
}

// connectLocked 内部连接逻辑，调用方需持有写锁
func (p *Publisher) connectLocked() error {
	if p.cfg.URL == "" {
		return errors.New("rabbitmq url is empty")
	}

	conn, err := amqp.Dial(p.cfg.URL)
	if err != nil {
		return err
	}
	ch, err := conn.Channel()
	if err != nil {
		_ = conn.Close()
		return err
	}

	exchange := p.resolveExchange()
	exchangeType := p.resolveExchangeType()
	if err := ch.ExchangeDeclare(exchange, exchangeType, true, false, false, false, nil); err != nil {
		_ = ch.Close()
		_ = conn.Close()
		return err
	}

	p.conn = conn
	p.ch = ch

	// 启动断开监听，触发自动重连
	p.watchConnection(conn)
	return nil
}

// watchConnection 监听连接关闭事件，触发重连
func (p *Publisher) watchConnection(conn *amqp.Connection) {
	connCloseCh := conn.NotifyClose(make(chan *amqp.Error, 1))

	p.reconnWg.Add(1)
	go func() {
		defer p.reconnWg.Done()
		select {
		case amqpErr, ok := <-connCloseCh:
			if !ok || p.closed.Load() {
				return
			}
			p.logger.Warn("RabbitMQ 连接断开，开始自动重连", zap.Error(amqpErr))
			p.handleDisconnect()
			p.reconnectLoop()
		case <-p.closeCh:
			return
		}
	}()
}

// handleDisconnect 清理已断开的连接资源
func (p *Publisher) handleDisconnect() {
	p.mu.Lock()
	defer p.mu.Unlock()

	// 尝试关闭残留资源（忽略错误，因为连接可能已经断开）
	if p.ch != nil {
		_ = p.ch.Close()
		p.ch = nil
	}
	if p.conn != nil {
		_ = p.conn.Close()
		p.conn = nil
	}
}

// reconnectLoop 指数退避重连循环
func (p *Publisher) reconnectLoop() {
	cfg := p.reconnectConfig()
	attempt := 0

	for {
		if p.closed.Load() {
			return
		}
		if cfg.maxRetries > 0 && attempt >= cfg.maxRetries {
			p.logger.Error("RabbitMQ 重连已达最大重试次数，放弃重连",
				zap.Int("maxRetries", cfg.maxRetries))
			return
		}

		interval := p.backoffInterval(attempt, cfg.initialInterval, cfg.maxInterval)
		attempt++

		p.logger.Info("RabbitMQ 重连等待中",
			zap.Int("attempt", attempt),
			zap.Duration("interval", interval))

		select {
		case <-time.After(interval):
		case <-p.closeCh:
			return
		}

		p.mu.Lock()
		err := p.connectLocked()
		p.mu.Unlock()

		if err == nil {
			p.logger.Info("RabbitMQ 重连成功", zap.Int("attempt", attempt))
			return
		}
		p.logger.Warn("RabbitMQ 重连失败",
			zap.Int("attempt", attempt),
			zap.Error(err))
	}
}

type reconnectCfg struct {
	maxRetries      int
	initialInterval time.Duration
	maxInterval     time.Duration
}

func (p *Publisher) reconnectConfig() reconnectCfg {
	cfg := reconnectCfg{
		maxRetries:      p.cfg.ReconnectMaxRetries,
		initialInterval: p.cfg.ReconnectInitialInterval,
		maxInterval:     p.cfg.ReconnectMaxInterval,
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
func (p *Publisher) backoffInterval(attempt int, initial, max time.Duration) time.Duration {
	backoff := time.Duration(float64(initial) * math.Pow(2, float64(attempt)))
	if backoff > max {
		backoff = max
	}
	return backoff
}

// Publish 发布消息，若 channel 不可用则尝试一次重连
func (p *Publisher) Publish(ctx context.Context, routingKey string, body []byte) error {
	if p.closed.Load() {
		return ErrPublisherClosed
	}

	ch := p.getChannel()
	if ch == nil {
		// channel 为空说明正在重连中，尝试等待一次重连完成后重试
		if err := p.waitAndReconnect(ctx); err != nil {
			return err
		}
		ch = p.getChannel()
		if ch == nil {
			return errors.New("rabbitmq channel not available after reconnect")
		}
	}

	traceID := trace.GetTraceID(ctx)
	traceParent := trace.TraceParentFromContext(ctx)

	// 构建 AMQP Headers
	headers := amqp.Table{}
	if traceID != "" {
		headers[trace.HeaderName] = traceID
	}
	if traceParent != "" {
		headers[trace.HeaderNameTraceParent] = traceParent
	}

	exchange := p.resolveExchange()
	err := ch.PublishWithContext(ctx, exchange, routingKey, false, false, amqp.Publishing{
		ContentType:  "application/json",
		Body:         body,
		DeliveryMode: amqp.Persistent,
		Headers:      headers, // 注入 TraceID
	})
	if err != nil {
		// 发布失败，可能 channel 已断开（会由 watchConnection 触发重连）
		p.logger.Warn("RabbitMQ 发布失败", zap.String("routingKey", routingKey), zap.Error(err))
		return err
	}
	return nil
}

// waitAndReconnect 在 Publish 时发现 channel 不可用，等待短暂时间看重连是否完成
func (p *Publisher) waitAndReconnect(ctx context.Context) error {
	// 最多等待 5s，期间重连协程可能已经恢复连接
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	timeout := time.After(5 * time.Second)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-p.closeCh:
			return ErrPublisherClosed
		case <-timeout:
			return errors.New("rabbitmq reconnect timeout, channel not available")
		case <-ticker.C:
			if p.getChannel() != nil {
				return nil
			}
		}
	}
}

func (p *Publisher) getChannel() *amqp.Channel {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.ch
}

// Close 关闭连接并停止重连协程
func (p *Publisher) Close() error {
	if !p.closed.CompareAndSwap(false, true) {
		return nil // 已经关闭
	}
	close(p.closeCh)

	// 等待重连协程退出
	p.reconnWg.Wait()

	p.mu.Lock()
	defer p.mu.Unlock()

	var err error
	if p.ch != nil {
		err = p.ch.Close()
		p.ch = nil
	}
	if p.conn != nil {
		closeErr := p.conn.Close()
		if err == nil {
			err = closeErr
		}
		p.conn = nil
	}
	return err
}

func (p *Publisher) resolveExchange() string {
	if p.cfg.Exchange != "" {
		return p.cfg.Exchange
	}
	return "dehaze.tasks"
}

func (p *Publisher) resolveExchangeType() string {
	if p.cfg.ExchangeType != "" {
		return p.cfg.ExchangeType
	}
	return "direct"
}
