package dept

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// IDeptRepository 部门仓储接口
type IDeptRepository interface {
	// FindByID 根据 ID 查询部门
	FindByID(ctx context.Context, id int64) (*model.SysDept, error)

	// FindAll 查询所有部门
	FindAll(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error)

	// FindByParentID 根据父 ID 查询子部门
	FindByParentID(ctx context.Context, parentID int64) ([]model.SysDept, error)

	// FindIDByName 根据部门名称查询 ID
	FindIDByName(ctx context.Context, name string) (int64, error)

	// Create 创建部门
	Create(ctx context.Context, dept *model.SysDept) error

	// Update 更新部门
	Update(ctx context.Context, dept *model.SysDept) error

	// Delete 删除部门（支持批量软删除）
	Delete(ctx context.Context, ids []int64) error

	// HasChildren 检查部门是否有子部门
	HasChildren(ctx context.Context, id int64) (bool, error)

	// HasUsers 检查部门是否关联用户
	HasUsers(ctx context.Context, deptID int64) (bool, error)

	// GetOptions 获取部门下拉选项
	GetOptions(ctx context.Context) ([]read.Option, error)

	// GetFormData 获取部门表单数据
	GetFormData(ctx context.Context, deptID int64) (*bo.DeptFormBO, error)

	// GetSubDeptIDs 获取部门及所有子部门 ID
	GetSubDeptIDs(ctx context.Context, deptID int64) ([]int64, error)
}
