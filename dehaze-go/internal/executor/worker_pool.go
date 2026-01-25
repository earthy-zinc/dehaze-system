package executor

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"go.uber.org/zap"
)

// TaskFunc 任务执行函数类型
type TaskFunc func(ctx context.Context) error

// TaskContext 任务上下文
type TaskContext struct {
	ID           string             // 任务ID
	TaskType     string             // 任务类型
	TotalFiles   int                // 总文件数
	Processed    atomic.Int32       // 已处理文件数
	CancelFunc   context.CancelFunc // 取消函数
	ProgressChan chan int           // 进度通道
	ResultChan   chan *TaskResult   // 结果通道
	mu           sync.RWMutex       // 互斥锁
	status       string             // 当前状态
	errorMessage string             // 错误信息
	startedAt    time.Time          // 开始时间
	completedAt  time.Time          // 完成时间
}

// TaskResult 任务结果
type TaskResult struct {
	Type         string // 任务类型
	Status       string // 任务状态: pending, processing, completed, failed, cancelled
	Progress     int    // 进度(0-100)
	Processed    int    // 已处理文件数
	TotalFiles   int    // 总文件数
	Result       string // 结果数据
	ErrorMessage string // 错误信息
}

// NewTaskContext 创建任务上下文
func NewTaskContext(id, taskType string) *TaskContext {
	_, cancel := context.WithCancel(context.Background())
	return &TaskContext{
		ID:           id,
		TaskType:     taskType,
		status:       "pending",
		CancelFunc:   cancel,
		ProgressChan: make(chan int, 10),
		ResultChan:   make(chan *TaskResult, 1),
		startedAt:    time.Now(),
	}
}

// GetStatus 获取任务状态
func (tc *TaskContext) GetStatus() string {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	return tc.status
}

// SetStatus 设置任务状态
func (tc *TaskContext) SetStatus(status string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	tc.status = status
	if status == "completed" || status == "failed" || status == "cancelled" {
		tc.completedAt = time.Now()
	}
}

// GetErrorMessage 获取错误信息
func (tc *TaskContext) GetErrorMessage() string {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	return tc.errorMessage
}

// SetErrorMessage 设置错误信息
func (tc *TaskContext) SetErrorMessage(msg string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	tc.errorMessage = msg
}

// UpdateProgress 更新进度
func (tc *TaskContext) UpdateProgress(processed int) {
	tc.Processed.Store(int32(processed))
	if tc.TotalFiles > 0 {
		progress := processed * 100 / tc.TotalFiles
		select {
		case tc.ProgressChan <- progress:
		default:
			// 通道已满，丢弃进度更新
		}
	}
}

// Cancel 取消任务
func (tc *TaskContext) Cancel() {
	tc.CancelFunc()
	tc.SetStatus("cancelled")
}

// GetStartedAt 获取开始时间
func (tc *TaskContext) GetStartedAt() time.Time {
	return tc.startedAt
}

// GetCompletedAt 获取完成时间
func (tc *TaskContext) GetCompletedAt() time.Time {
	return tc.completedAt
}

// Close 关闭任务上下文
func (tc *TaskContext) Close() {
	close(tc.ProgressChan)
	close(tc.ResultChan)
}

// WorkerPool Worker池
type WorkerPool struct {
	name         string
	coreSize     int
	maxSize      int
	queueSize    int
	taskTimeout  time.Duration
	shutdownWait time.Duration

	taskCh      chan TaskFunc
	taskCtxMap  sync.Map // taskID -> *TaskContext
	workerCount atomic.Int32
	wg          sync.WaitGroup
	closeCh     chan struct{}
	once        sync.Once
}

// NewWorkerPool 创建Worker池
func NewWorkerPool(name string, cfg options.WorkerPool) *WorkerPool {
	return &WorkerPool{
		name:         name,
		coreSize:     cfg.CorePoolSize,
		maxSize:      cfg.MaxPoolSize,
		queueSize:    cfg.QueueSize,
		taskTimeout:  time.Duration(cfg.TaskTimeout) * time.Second,
		shutdownWait: time.Duration(cfg.ShutdownWait) * time.Second,
		taskCh:       make(chan TaskFunc, cfg.QueueSize),
		closeCh:      make(chan struct{}),
	}
}

// Start 启动Worker池
func (p *WorkerPool) Start() {
	// 启动核心Worker
	for i := 0; i < p.coreSize; i++ {
		p.newWorker(fmt.Sprintf("%s-%d", p.name, i))
	}
	// 启动动态扩容监控
	go p.monitor()
}

// Stop 停止Worker池
func (p *WorkerPool) Stop() {
	p.once.Do(func() {
		close(p.closeCh)
		// 等待所有Worker完成任务
		done := make(chan struct{})
		go func() {
			p.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			logger.Info("Worker池正常关闭", zap.String("pool", p.name))
		case <-time.After(p.shutdownWait):
			logger.Warn("Worker池强制关闭", zap.String("pool", p.name))
		}
	})
}

// newWorker 创建新Worker
func (p *WorkerPool) newWorker(name string) {
	p.wg.Add(1)
	p.workerCount.Add(1)
	go func() {
		defer p.wg.Done()
		defer p.workerCount.Add(-1)
		logger.Info("Worker启动", zap.String("pool", p.name), zap.String("worker", name))
		for {
			select {
			case task := <-p.taskCh:
				p.executeTask(task, name)
			case <-p.closeCh:
				logger.Info("Worker退出", zap.String("pool", p.name), zap.String("worker", name))
				return
			}
		}
	}()
}

// executeTask 执行任务
func (p *WorkerPool) executeTask(task TaskFunc, workerName string) {
	defer func() {
		if r := recover(); r != nil {
			logger.Error("Task panic", zap.String("pool", p.name), zap.String("worker", workerName), zap.Any("error", r))
		}
	}()

	// 设置超时上下文
	ctx, cancel := context.WithTimeout(context.Background(), p.taskTimeout)
	defer cancel()

	// 执行任务
	err := task(ctx)
	if err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("Task执行失败",
			zap.String("pool", p.name),
			zap.String("worker", workerName),
			zap.Error(err))
	}
}

// monitor 监控队列长度和动态扩容
func (p *WorkerPool) monitor() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			queueLen := len(p.taskCh)
			currentWorkers := int(p.workerCount.Load())

			// 队列积压严重，扩容
			if queueLen > p.queueSize/2 && currentWorkers < p.maxSize {
				newWorkerCount := p.coreSize + (queueLen-p.queueSize/2)/10
				if newWorkerCount > p.maxSize {
					newWorkerCount = p.maxSize
				}
				for i := currentWorkers; i < newWorkerCount; i++ {
					p.newWorker(fmt.Sprintf("%s-dynamic-%d", p.name, i))
				}
			}
		case <-p.closeCh:
			return
		}
	}
}

// Submit 提交任务
func (p *WorkerPool) Submit(taskFunc TaskFunc) error {
	select {
	case p.taskCh <- taskFunc:
		return nil
	case <-p.closeCh:
		return errors.New("Worker池已关闭")
	default:
		return errors.New("任务队列已满")
	}
}

// SubmitWithCtx 提交任务并返回任务上下文
func (p *WorkerPool) SubmitWithCtx(taskID string, taskType string, totalFiles int, taskFunc TaskFunc) (*TaskContext, error) {
	taskCtx := NewTaskContext(taskID, taskType)
	taskCtx.TotalFiles = totalFiles
	p.taskCtxMap.Store(taskID, taskCtx)

	// 包装任务函数来处理进度更新
	wrappedFunc := func(ctx context.Context) error {
		taskCtx.SetStatus("processing")
		if err := taskFunc(ctx); err != nil {
			taskCtx.SetStatus("failed")
			taskCtx.SetErrorMessage(err.Error())
			taskCtx.ResultChan <- &TaskResult{
				Type:         taskType,
				Status:       "failed",
				ErrorMessage: err.Error(),
			}
			return err
		}
		taskCtx.SetStatus("completed")
		taskCtx.ResultChan <- &TaskResult{
			Type:   taskType,
			Status: "completed",
		}
		return nil
	}

	if err := p.Submit(wrappedFunc); err != nil {
		p.taskCtxMap.Delete(taskID)
		taskCtx.Close()
		return nil, err
	}

	return taskCtx, nil
}

// GetTaskContext 获取任务上下文
func (p *WorkerPool) GetTaskContext(taskID string) (*TaskContext, bool) {
	if ctx, ok := p.taskCtxMap.Load(taskID); ok {
		return ctx.(*TaskContext), true
	}
	return nil, false
}

// CancelTask 取消任务
func (p *WorkerPool) CancelTask(taskID string) bool {
	if ctx, ok := p.taskCtxMap.Load(taskID); ok {
		taskCtx := ctx.(*TaskContext)
		taskCtx.Cancel()
		return true
	}
	return false
}

// RemoveTaskContext 移除任务上下文
func (p *WorkerPool) RemoveTaskContext(taskID string) {
	if ctx, ok := p.taskCtxMap.Load(taskID); ok {
		taskCtx := ctx.(*TaskContext)
		taskCtx.Close()
		p.taskCtxMap.Delete(taskID)
	}
}

// GetStatus 获取Worker池状态
func (p *WorkerPool) GetStatus() map[string]interface{} {
	return map[string]interface{}{
		"name":          p.name,
		"workerCount":   p.workerCount.Load(),
		"queueLength":   len(p.taskCh),
		"queueCapacity": p.queueSize,
		"coreSize":      p.coreSize,
		"maxSize":       p.maxSize,
	}
}
