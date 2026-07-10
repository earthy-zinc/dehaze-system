package role

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ====================
// 角色管理服务接口
// ====================

// IRoleService 角色服务接口
type IRoleService interface {
	// GetPage 角色分页列表
	GetPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error)

	// GetOptions 获取角色下拉选项（isRoot 为 false 时排除 ROOT 角色）
	GetOptions(ctx context.Context, isRoot bool) ([]vo.Option, error)

	// GetFormData 获取角色表单数据
	GetFormData(ctx context.Context, id int64) (*bo.RoleFormBO, error)

	// Create 创建角色
	Create(ctx context.Context, form *bo.RoleFormBO) error

	// Update 更新角色
	Update(ctx context.Context, id int64, form *bo.RoleFormBO) error

	// Delete 删除角色（支持批量）
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新角色状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// GetMenuIDs 获取角色菜单 ID 集合
	GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error)

	// AssignMenus 分配菜单权限
	AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error

	// GetMaximumDataScope 获取最大范围的数据权限
	GetMaximumDataScope(ctx context.Context, roles []string) (dataScope *int8, err error)
}
