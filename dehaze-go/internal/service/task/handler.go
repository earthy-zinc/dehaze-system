package task

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"go.uber.org/zap"
)

// HandleExportTaskMessage 处理导出任务消息的 MQ Handler
// Go 端暂未实现任务执行策略，收到消息后将任务标记为 FAILED
// 待后续实现 TaskStrategy 后替换为真实执行逻辑
func (ts *TaskService) HandleExportTaskMessage(ctx context.Context, body []byte) error {
	var msg TaskMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		ts.logger.Error("解析任务消息失败，丢弃消息",
			zap.Error(err))
		return nil // 返回 nil 使消息 Ack，避免无限重试格式错误的消息
	}

	ts.logger.Info("收到导出任务消息",
		zap.String("taskID", msg.TaskID),
		zap.String("taskType", msg.TaskType),
		zap.String("traceID", msg.TraceID))

	// 更新任务状态为 PROCESSING
	if err := ts.UpdateTaskStatus(ctx, msg.TaskID, model.TaskStatusProcessing, ""); err != nil {
		ts.logger.Warn("更新任务状态为 PROCESSING 失败",
			zap.String("taskID", msg.TaskID),
			zap.Error(err))
		return err
	}

	// Go 端未实现任务执行策略，标记为 FAILED
	// 待 TaskStrategy 实现后替换为真实执行逻辑
	errMsg := "Go 后端暂未实现任务执行策略，请联系管理员使用 Java 或 Python 后端执行任务"
	if err := ts.UpdateTaskStatus(ctx, msg.TaskID, model.TaskStatusFailed, errMsg); err != nil {
		ts.logger.Warn("更新任务状态为 FAILED 失败",
			zap.String("taskID", msg.TaskID),
			zap.Error(err))
		return err
	}

	// 更新 MQ 重试次数（从消息 header 读取，若有）
	if retryCount, ok := msg.Extra["retryCount"]; ok {
		if rc, ok := retryCount.(float64); ok {
			_ = ts.UpdateRetryCount(ctx, msg.TaskID, int(rc))
		}
	}

	return nil
}

// HandleDLQMessage 处理死信队列消息的 Handler
// 将死信任务标记为 FAILED 并记录错误信息
func (ts *TaskService) HandleDLQMessage(ctx context.Context, body []byte, headers map[string]interface{}) error {
	var msg TaskMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		ts.logger.Error("解析死信消息失败",
			zap.Error(err))
		return nil
	}

	ts.logger.Warn("任务进入死信队列",
		zap.String("taskID", msg.TaskID),
		zap.String("taskType", msg.TaskType),
		zap.Any("headers", headers))

	errMsg := fmt.Sprintf("任务经过 %d 次重试后仍失败，已进入死信队列", mq.MaxRetryCount)
	if err := ts.UpdateTaskStatus(ctx, msg.TaskID, model.TaskStatusFailed, errMsg); err != nil {
		ts.logger.Warn("更新死信任务状态失败",
			zap.String("taskID", msg.TaskID),
			zap.Error(err))
		return err
	}

	return nil
}
