package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"github.com/earthyzinc/dehaze-go/pkg/trace"
	"go.uber.org/zap"
)

// RabbitMQTaskExecutor RabbitMQ 异步任务执行器（仅发布消息）
type RabbitMQTaskExecutor struct {
	cfg       options.RabbitMQ
	publisher *mq.Publisher
	logger    *zap.Logger
}

// NewRabbitMQTaskExecutor 创建 RabbitMQ 任务执行器
func NewRabbitMQTaskExecutor(cfg options.RabbitMQ, logger *zap.Logger) *RabbitMQTaskExecutor {
	return &RabbitMQTaskExecutor{
		cfg:       cfg,
		publisher: mq.NewPublisher(cfg, logger),
		logger:    logger,
	}
}

// Initialize 初始化 RabbitMQ 连接
func (e *RabbitMQTaskExecutor) Initialize() error {
	if !e.cfg.Enabled {
		return errors.New("rabbitmq is disabled")
	}
	return e.publisher.Connect()
}

// Shutdown 关闭连接
func (e *RabbitMQTaskExecutor) Shutdown() error {
	return e.publisher.Close()
}

// PublishExportTask 发布导出任务
func (e *RabbitMQTaskExecutor) PublishExportTask(ctx context.Context, taskID int64, form bo.ExportTaskCreateForm) error {
	msg := TaskMessage{
		TaskID:    fmt.Sprintf("%d", taskID),
		TaskType:  "export",
		Payload:   form,
		CreatedAt: time.Now(),
	}
	return e.PublishTask(ctx, msg)
}

// PublishTask 发布通用任务消息
func (e *RabbitMQTaskExecutor) PublishTask(ctx context.Context, msg TaskMessage) error {
	if msg.TaskID == "" {
		return errors.New("taskId is empty")
	}
	if msg.TaskType == "" {
		return errors.New("taskType is empty")
	}
	if msg.CreatedAt.IsZero() {
		msg.CreatedAt = time.Now()
	}
	if msg.TraceID == "" {
		msg.TraceID = trace.GetTraceID(ctx)
	}

	routingKey := e.buildRoutingKey(msg.TaskType)
	body, err := json.Marshal(msg)
	if err != nil {
		return err
	}

	err = e.publisher.Publish(ctx, routingKey, body)
	if err != nil {
		e.logger.Error("发布任务消息失败", zap.String("taskID", msg.TaskID), zap.String("taskType", msg.TaskType), zap.Error(err))
	}
	return err
}

func (e *RabbitMQTaskExecutor) buildRoutingKey(taskType string) string {
	prefix := e.cfg.RoutingKeyPrefix
	if prefix == "" {
		prefix = "task"
	}
	return fmt.Sprintf("%s.%s", prefix, taskType)
}
