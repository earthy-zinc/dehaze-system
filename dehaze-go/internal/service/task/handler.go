package task

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/pkg/mq"
	"go.uber.org/zap"
)

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
		zap.Int64("dbTaskID", msg.DbTaskID),
		zap.String("taskID", msg.TaskID),
		zap.String("taskType", msg.TaskType),
		zap.Any("headers", headers))

	errMsg := fmt.Sprintf("任务经过 %d 次重试后仍失败，已进入死信队列", mq.MaxRetryCount)
	// UpdateTaskStatus 内部通过 UUID (task_id) 查询，使用 msg.TaskID
	if err := ts.UpdateTaskStatus(ctx, msg.TaskID, model.TaskStatusFailed, errMsg); err != nil {
		ts.logger.Warn("更新死信任务状态失败",
			zap.Int64("dbTaskID", msg.DbTaskID),
			zap.String("taskID", msg.TaskID),
			zap.Error(err))
		return err
	}

	return nil
}
