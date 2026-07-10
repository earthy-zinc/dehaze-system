package role

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// ====================
// 角色管理 Repository
// ====================

// IRoleRepository 角色仓储接口
type IRoleRepository interface {
	// FindByID 根据 ID 查询角色
	FindByID(ctx context.Context, id int64) (*model.SysRole, error)

	// FindByIDs 根据 ID 列表批量查询角色
	FindByIDs(ctx context.Context, ids []int64) ([]*model.SysRole, error)

	// FindByCode 根据编码查询角色
	FindByCode(ctx context.Context, code string) (*model.SysRole, error)

	// ExistsByCode 检查角色编码是否存在
	ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error)

	// ExistsByName 检查角色名称是否存在
	ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error)

	// FindPage 分页查询角色列表
	FindPage(ctx context.Context, q *query.RolePageQuery) (*read.PageResult[read.RolePage], error)

	// FindOptions 获取角色下拉选项
	FindOptions(ctx context.Context) ([]read.Option, error)

	// Create 创建角色
	Create(ctx context.Context, role *model.SysRole) error

	// Update 更新角色
	Update(ctx context.Context, role *model.SysRole) error

	// UpdateStatus 更新角色状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// Delete 删除角色（逻辑删除）
	Delete(ctx context.Context, ids []int64) error

	// HasUsers 检查角色是否关联用户
	HasUsers(ctx context.Context, roleID int64) (bool, error)

	// HasUsersInBatch 批量检查角色是否关联用户，返回关联用户的角色ID集合
	HasUsersInBatch(ctx context.Context, roleIDs []int64) (map[int64]bool, error)

	// GetMenuIDs 获取角色菜单 ID 列表
	GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error)

	// AssignMenus 分配角色菜单
	AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error

	// DeleteMenusByRoleIDs 批量删除角色的菜单关联
	DeleteMenusByRoleIDs(ctx context.Context, roleIDs []int64) error

	// DeleteWithMenus 删除角色及其菜单关联（事务）
	DeleteWithMenus(ctx context.Context, roleIDs []int64) error

	// GetFormData 获取角色表单数据
	GetFormData(ctx context.Context, roleID int64) (*read.RoleForm, error)

	// GetMinimumDataScope 获取角色的最小数据权限范围
	GetMinimumDataScope(ctx context.Context, roleCodes []string) (*int8, error)
}
