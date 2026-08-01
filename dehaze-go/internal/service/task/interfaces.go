package task

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ITaskService 任务服务接口
type ITaskService interface {
	// GetPage 任务分页列表
	GetPage(ctx context.Context, q *query.TaskPageQuery) (*vo.PageResult[vo.TaskVO], error)

	// GetByID 根据 ID 获取任务详情
	GetByID(ctx context.Context, id int64) (*vo.TaskDetailVO, error)

	// Create 创建任务
	Create(ctx context.Context, form *bo.TaskBO) (int64, error)

	// Delete 删除任务
	Delete(ctx context.Context, ids []int64) error

	// Cancel 取消任务
	Cancel(ctx context.Context, id int64) error
}

// TaskMessage 异步任务消息（三端统一契约：最小自描述 JSON）
type TaskMessage struct {
	DbTaskID int64  `json:"db_task_id"`
	TaskID   string `json:"task_id"`
	TaskType string `json:"task_type"`
}

// AsyncTaskExecutor 异步任务执行接口（RabbitMQ 实现）
type AsyncTaskExecutor interface {
	Initialize() error
	Shutdown() error
	IsConnected() bool
	PublishTask(ctx context.Context, msg TaskMessage) error
}
