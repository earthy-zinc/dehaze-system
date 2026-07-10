package task

import (
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"go.uber.org/zap"
)

// NewAsyncTaskExecutor 创建异步任务执行器
// 目前默认使用 RabbitMQ 实现，后续可在此扩展更多实现
func NewAsyncTaskExecutor(cfg options.RabbitMQ, logger *zap.Logger) AsyncTaskExecutor {
	return NewRabbitMQTaskExecutor(cfg, logger)
}
