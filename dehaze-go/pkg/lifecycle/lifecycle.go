// Package lifecycle 管理应用级 context 与关键异步 goroutine 的生命周期
// shutdown 时取消 context 并等待受管理 goroutine 完成，避免静默丢数据
package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Manager 持有应用级 context 与 WaitGroup
type Manager struct {
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewManager() *Manager {
	ctx, cancel := context.WithCancel(context.Background())
	return &Manager{ctx: ctx, cancel: cancel}
}

// Context 返回应用级 context，shutdown 时被取消
func (m *Manager) Context() context.Context { return m.ctx }

// Go 启动一个受管理的 goroutine，shutdown 时会等待其完成
func (m *Manager) Go(fn func(ctx context.Context)) {
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		fn(m.ctx)
	}()
}

// Shutdown 取消 context 并等待所有受管理 goroutine 完成（带超时）
func (m *Manager) Shutdown(timeout time.Duration) error {
	m.cancel()
	done := make(chan struct{})
	go func() {
		m.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-time.After(timeout):
		return fmt.Errorf("等待异步任务完成超时（%s），可能有任务未完成", timeout)
	}
}
