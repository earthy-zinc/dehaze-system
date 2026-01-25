package protection

import (
	"context"
	"sync"
)

// call 表示一个正在进行或已完成的调用
type call struct {
	wg   sync.WaitGroup
	val  any
	err  error
	done bool
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
// 对于相同的key，同时发起的多个请求只会执行一次fn，其他请求等待并共享结果
func (sf *SingleFlight) Do(ctx context.Context, key string, fn func() (any, error)) (any, error) {
	sf.mu.Lock()

	// 检查是否已有相同key的调用在进行中
	if c, ok := sf.calls[key]; ok {
		sf.mu.Unlock()
		// 等待已有调用完成
		c.wg.Wait()
		return c.val, c.err
	}

	// 创建新的调用
	c := &call{}
	c.wg.Add(1)
	sf.calls[key] = c
	sf.mu.Unlock()

	// 执行实际操作
	c.val, c.err = sf.doCall(ctx, fn)
	c.done = true
	c.wg.Done()

	// 清理
	sf.mu.Lock()
	delete(sf.calls, key)
	sf.mu.Unlock()

	return c.val, c.err
}

// doCall 执行实际的调用，支持context取消
func (sf *SingleFlight) doCall(ctx context.Context, fn func() (any, error)) (any, error) {
	done := make(chan struct{})
	var val any
	var err error

	go func() {
		val, err = fn()
		close(done)
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-done:
		return val, err
	}
}

// Forget 删除指定key的调用记录
// 允许相同key的后续请求重新执行fn
func (sf *SingleFlight) Forget(key string) {
	sf.mu.Lock()
	delete(sf.calls, key)
	sf.mu.Unlock()
}
