package task

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/database"
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

// IsConnected 返回 Publisher 连接是否活跃
func (e *RabbitMQTaskExecutor) IsConnected() bool {
	return e.publisher.IsConnected()
}

// PublishTask 发布通用任务消息
// 使用分布式锁防止同一 taskID 被重复发布
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
	if msg.CreatedBy == 0 {
		if userID := database.GetUserID(ctx); userID > 0 {
			msg.CreatedBy = userID
		}
	}

	// 分布式锁：防止同一 taskID 被并发重复发布
	lockKey := "task:publish:" + msg.TaskID
	cacheClient := cache.GetCache()
	if cacheClient != nil {
		token, acquired, lockErr := cacheClient.Lock(ctx, lockKey, 30*time.Second)
		if lockErr != nil {
			e.logger.Warn("获取任务发布锁失败，降级放行", zap.String("taskID", msg.TaskID), zap.Error(lockErr))
		} else if !acquired {
			return fmt.Errorf("任务 %s 正在发布中，请勿重复提交", msg.TaskID)
		} else {
			defer cacheClient.Unlock(ctx, lockKey, token)
		}
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
