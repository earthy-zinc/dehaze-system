package algorithm

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// ====================
// 算法管理 Repository
// ====================

// IAlgorithmRepository 算法仓储接口
type IAlgorithmRepository interface {
	// FindByID 根据 ID 查询算法
	FindByID(ctx context.Context, id int64) (*model.SysAlgorithm, error)

	// FindPage 分页查询算法
	FindPage(ctx context.Context, q *query.AlgorithmQuery) (*read.PageResult[read.Algorithm], error)

	// FindAll 查询所有算法（用于树形列表）
	FindAll(ctx context.Context, q *query.AlgorithmQuery) ([]read.Algorithm, error)

	// FindOptions 获取算法下拉选项
	FindOptions(ctx context.Context) ([]read.Option, error)

	// Create 创建算法
	Create(ctx context.Context, algorithm *model.SysAlgorithm) error

	// Update 更新算法
	Update(ctx context.Context, algorithm *model.SysAlgorithm) error

	// Delete 删除算法
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新算法状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// FindVersionsByAlgorithmID 查询算法版本历史（按创建时间降序）
	FindVersionsByAlgorithmID(ctx context.Context, algorithmID int64) ([]model.SysAlgorithmVersion, error)
}
