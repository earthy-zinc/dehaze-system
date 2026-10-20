package protection

import (
	"context"
	"sync"
)

// call 表示一个正在进行或已完成的调用
type call struct {
	val  any
	err  error
	done chan struct{}
}

// SingleFlight 实现请求合并，防止缓存击穿
type SingleFlight struct {
	mu    sync.Mutex
	calls map[string]*call
}

// NewSingleFlight 创建 SingleFlight 实例
func NewSingleFlight() *SingleFlight {
	return &SingleFlight{
		calls: make(map[string]*call),
	}
}

// Do 执行带去重的操作
// 对于相同的key，同时发起的多个请求只会执行一次fn，其他请求等待并共享结果。
// 某个等待者的 ctx 取消时仅影响该等待者自身，不影响 fn 的执行和其他等待者。
func (sf *SingleFlight) Do(ctx context.Context, key string, fn func() (any, error)) (any, error) {
	sf.mu.Lock()

	// 检查是否已有相同key的调用在进行中
	if c, ok := sf.calls[key]; ok {
		sf.mu.Unlock()
		// 等待已有调用完成，支持当前 ctx 取消
		select {
		case <-c.done:
			return c.val, c.err
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// 创建新的调用
	c := &call{done: make(chan struct{})}
	sf.calls[key] = c
	sf.mu.Unlock()

	// 同步执行实际操作（不启动额外 goroutine，避免泄漏）
	c.val, c.err = fn()
	close(c.done)

	// 清理
	sf.mu.Lock()
	delete(sf.calls, key)
	sf.mu.Unlock()

	return c.val, c.err
}

// Forget 删除指定key的调用记录
// 允许相同key的后续请求重新执行fn
func (sf *SingleFlight) Forget(key string) {
	sf.mu.Lock()
	delete(sf.calls, key)
	sf.mu.Unlock()
}
