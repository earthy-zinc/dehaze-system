package algorithm

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ====================
// 算法管理服务接口
// ====================

// IAlgorithmService 算法服务接口
type IAlgorithmService interface {
	// GetPage 算法分页列表
	GetPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error)

	// GetTree 获取算法树形列表
	GetTree(ctx context.Context, q *query.AlgorithmQuery) ([]vo.AlgorithmVO, error)

	// GetOptions 获取算法下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetFormData 获取算法表单数据
	GetFormData(ctx context.Context, id int64) (*bo.AlgorithmFormBO, error)

	// Create 创建算法
	Create(ctx context.Context, form *bo.AlgorithmFormBO) error

	// Update 更新算法
	Update(ctx context.Context, id int64, form *bo.AlgorithmFormBO) error

	// Delete 删除算法
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新算法状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// Compare 批量查询算法用于对比
	Compare(ctx context.Context, ids []int64) ([]model.SysAlgorithm, error)
}
