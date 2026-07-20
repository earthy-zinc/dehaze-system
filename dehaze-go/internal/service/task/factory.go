package task

import (
	"fmt"

	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"go.uber.org/zap"
)

// ExecutorType 任务执行器类型
const (
	ExecutorTypeRabbitMQ = "rabbitmq"
)

// NewAsyncTaskExecutor 创建异步任务执行器（策略工厂）
// 根据 executorType 选择具体实现，当前仅支持 "rabbitmq"。
// 后续可扩展 "kafka"、"local" 等实现，只需在此 switch 中增加分支。
func NewAsyncTaskExecutor(cfg options.RabbitMQ, logger *zap.Logger, executorType ...string) AsyncTaskExecutor {
	typ := ExecutorTypeRabbitMQ
	if len(executorType) > 0 && executorType[0] != "" {
		typ = executorType[0]
	}

	switch typ {
	case ExecutorTypeRabbitMQ:
		return NewRabbitMQTaskExecutor(cfg, logger)
	default:
		logger.Warn(fmt.Sprintf("未知的任务执行器类型 %q，降级使用 RabbitMQ", typ))
		return NewRabbitMQTaskExecutor(cfg, logger)
	}
}
