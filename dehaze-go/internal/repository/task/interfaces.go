package task

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// ITaskRepository 任务仓储接口
type ITaskRepository interface {
	// FindByID 根据 ID 查询任务
	FindByID(ctx context.Context, id int64) (*model.SysTask, error)

	// FindByTaskID 根据任务唯一 ID 查询任务
	FindByTaskID(ctx context.Context, taskID string) (*model.SysTask, error)

	// FindByIdempotencyKey 根据客户端幂等键查询任务（用于去重）
	FindByIdempotencyKey(ctx context.Context, key string) (*model.SysTask, error)

	// FindPage 分页查询任务
	FindPage(ctx context.Context, q *query.TaskPageQuery) (*read.PageResult[read.Task], error)

	// Create 创建任务
	Create(ctx context.Context, task *model.SysTask) error

	// Update 更新任务
	Update(ctx context.Context, task *model.SysTask) error

	// UpdateFields 更新任务指定字段
	UpdateFields(ctx context.Context, id int64, fields map[string]interface{}) error

	// UpdateStatus 更新任务状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// Delete 删除任务
	Delete(ctx context.Context, ids []int64) error

	// UpdateExpiredTasks 更新过期任务状态
	UpdateExpiredTasks(ctx context.Context, threshold time.Time) (int64, error)

	// CountDatasetItems 统计数据集数据项数量
	CountDatasetItems(ctx context.Context, datasetID int64) (int64, error)

	// CountItemFiles 统计数据项文件数量
	CountItemFiles(ctx context.Context, itemIDs []int64) (int64, error)
}
