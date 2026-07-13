package dept

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// IDeptService 部门服务接口
type IDeptService interface {
	// GetList 获取部门列表
	GetList(ctx context.Context, q *query.DeptQuery) ([]vo.DeptVO, error)

	// GetFormData 获取部门表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DeptFormBO, error)

	// Create 创建部门
	Create(ctx context.Context, form *bo.DeptFormBO) (int64, error)

	// Update 更新部门
	Update(ctx context.Context, id int64, form *bo.DeptFormBO) error

	// Delete 删除部门（支持批量，级联删除子部门）
	Delete(ctx context.Context, ids []int64) error

	// GetOptions 获取部门下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)
}
