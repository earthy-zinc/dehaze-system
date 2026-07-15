package task

import (
	"context"
	"time"

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

// TaskMessage 异步任务消息
// Payload 为业务自定义内容，由消费者解析处理
// Total 用于进度估算，可为空
// CreatedAt 用于追踪与幂等处理
// TaskType 建议为: export/thumbnail/dataset 等
// TaskID 必须唯一
// TraceID 用于链路追踪
// CreatedBy 可用于审计
// Extra 预留扩展字段
// 注意: Payload 需要是可序列化对象
// 后续可按需加版本字段
type TaskMessage struct {
	TaskID    string         `json:"taskId"`
	TaskType  string         `json:"taskType"`
	Total     int            `json:"total,omitempty"`
	Payload   any            `json:"payload,omitempty"`
	TraceID   string         `json:"traceId,omitempty"`
	CreatedAt time.Time      `json:"createdAt"`
	CreatedBy int64          `json:"createdBy,omitempty"`
	Extra     map[string]any `json:"extra,omitempty"`
}

// AsyncTaskExecutor 异步任务执行接口（RabbitMQ 实现）
type AsyncTaskExecutor interface {
	Initialize() error
	Shutdown() error
	IsConnected() bool
	PublishExportTask(ctx context.Context, taskID string, form bo.ExportTaskCreateForm) error
	PublishTask(ctx context.Context, msg TaskMessage) error
}
