package protection

import (
	"errors"
	"sync"
	"time"
)

var (
	ErrCircuitOpen = errors.New("circuit breaker is open")
)

// Breaker 熔断器实现
type Breaker struct {
	mu              sync.RWMutex
	state           CircuitState
	failureCount    int
	successCount    int
	lastFailureTime time.Time
	lastStateChange time.Time

	// 配置
	failureThreshold int           // 失败次数阈值，达到后熔断
	successThreshold int           // 半开状态下成功次数阈值，达到后关闭熔断
	timeout          time.Duration // 熔断超时时间，超时后进入半开状态
	maxRequests      uint          // 半开状态下允许的最大请求数
	halfOpenCount    uint          // 半开状态下已处理的请求数
}

// BreakerOption 熔断器配置选项
type BreakerOption func(*Breaker)

// WithFailureThreshold 设置失败阈值
func WithFailureThreshold(threshold int) BreakerOption {
	return func(b *Breaker) {
		b.failureThreshold = threshold
	}
}

// WithTimeout 设置熔断超时时间
func WithTimeout(timeout time.Duration) BreakerOption {
	return func(b *Breaker) {
		b.timeout = timeout
	}
}

// WithMaxRequests 设置半开状态最大请求数
func WithMaxRequests(max uint) BreakerOption {
	return func(b *Breaker) {
		b.maxRequests = max
	}
}

// NewBreaker 创建熔断器
func NewBreaker(opts ...BreakerOption) *Breaker {
	b := &Breaker{
		state:            StateClosed,
		failureThreshold: 5,
		successThreshold: 3,
		timeout:          30 * time.Second,
		maxRequests:      3,
		lastStateChange:  time.Now(),
	}

	for _, opt := range opts {
		opt(b)
	}

	return b
}

// Execute 通过熔断器执行操作
func (b *Breaker) Execute(fn func() error) error {
	if err := b.beforeRequest(); err != nil {
		return err
	}

	err := fn()
	b.afterRequest(err)
	return err
}

// beforeRequest 请求前检查熔断器状态
func (b *Breaker) beforeRequest() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch b.state {
	case StateClosed:
		return nil
	case StateOpen:
		// 检查是否超时，超时则进入半开状态
		if time.Since(b.lastStateChange) >= b.timeout {
			b.toHalfOpen()
			return nil
		}
		return ErrCircuitOpen
	case StateHalfOpen:
		// 半开状态下限制请求数量
		if b.halfOpenCount >= b.maxRequests {
			return ErrCircuitOpen
		}
		b.halfOpenCount++
		return nil
	}
	return nil
}

// afterRequest 请求后更新熔断器状态
func (b *Breaker) afterRequest(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	switch {
	case IsBusiness(err):
		// 业务性错误（上游 4xx，如参数/权限拒绝）：上游是**可达且健康**的，
		// 按成功处理（不累计失败、并清零连续失败），否则业务负例连发会把熔断器打红。
		// 口径以 python 为准：python 调用侧把 HTTPStatusError 与 (Timeout, TransportError)
		// 分开处理，4xx 不参与任何熔断/冷却统计。
		b.onSuccess()
	case err != nil:
		b.onFailure()
	default:
		b.onSuccess()
	}
}

// BusinessError 标记"业务性错误"：错误由上游正常返回（4xx 业务拒绝），不代表上游不可用，
// 因而不参与熔断失败统计。
type BusinessError struct{ Err error }

func (e *BusinessError) Error() string { return e.Err.Error() }

func (e *BusinessError) Unwrap() error { return e.Err }

// MarkBusiness 包装业务性错误（nil 原样返回）
func MarkBusiness(err error) error {
	if err == nil {
		return nil
	}
	return &BusinessError{Err: err}
}

// IsBusiness 判断是否为业务性错误
func IsBusiness(err error) bool {
	var be *BusinessError
	return errors.As(err, &be)
}

func (b *Breaker) onSuccess() {
	switch b.state {
	case StateClosed:
		b.failureCount = 0
	case StateHalfOpen:
		b.successCount++
		if b.successCount >= b.successThreshold {
			b.toClosed()
		}
	}
}

func (b *Breaker) onFailure() {
	b.failureCount++
	b.lastFailureTime = time.Now()

	switch b.state {
	case StateClosed:
		if b.failureCount >= b.failureThreshold {
			b.toOpen()
		}
	case StateHalfOpen:
		b.toOpen()
	}
}

func (b *Breaker) toOpen() {
	b.state = StateOpen
	b.lastStateChange = time.Now()
	b.halfOpenCount = 0
	b.successCount = 0
}

func (b *Breaker) toHalfOpen() {
	b.state = StateHalfOpen
	b.lastStateChange = time.Now()
	b.halfOpenCount = 0
	b.successCount = 0
	b.failureCount = 0
}

func (b *Breaker) toClosed() {
	b.state = StateClosed
	b.lastStateChange = time.Now()
	b.failureCount = 0
	b.successCount = 0
	b.halfOpenCount = 0
}

// State 获取当前状态
func (b *Breaker) State() CircuitState {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.state
}

// Reset 手动重置熔断器
func (b *Breaker) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.toClosed()
}
