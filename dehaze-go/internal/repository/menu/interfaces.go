package menu

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// IMenuRepository 菜单仓储接口
type IMenuRepository interface {
	// FindByID 根据 ID 查询菜单
	FindByID(ctx context.Context, id int64) (*model.SysMenu, error)

	// FindAll 查询所有菜单
	FindAll(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error)

	// FindByParentID 根据父 ID 查询子菜单
	FindByParentID(ctx context.Context, parentID int64) ([]model.SysMenu, error)

	// Create 创建菜单
	Create(ctx context.Context, menu *model.SysMenu) error

	// Update 更新菜单
	Update(ctx context.Context, menu *model.SysMenu) error

	// Delete 删除菜单
	Delete(ctx context.Context, id int64) error

	// HasChildren 检查菜单是否有子菜单
	HasChildren(ctx context.Context, id int64) (bool, error)

	// ExistsByName 检查同级菜单名称是否存在
	// excludeID: 排除的菜单ID（用于更新时排除自身）
	ExistsByName(ctx context.Context, parentID int64, name string, excludeID int64) (bool, error)

	// ExistsByPath 检查同级菜单路径是否存在
	// excludeID: 排除的菜单ID（用于更新时排除自身）
	ExistsByPath(ctx context.Context, parentID int64, path string, excludeID int64) (bool, error)

	// ExistsByPerm 检查权限标识是否存在（全局唯一）
	// excludeID: 排除的菜单ID（用于更新时排除自身）
	ExistsByPerm(ctx context.Context, perm string, excludeID int64) (bool, error)

	// FindRoutesByRoles 根据角色获取路由菜单
	FindRoutesByRoles(ctx context.Context, roles []string) ([]model.SysMenu, error)

	// FindPermsByRoles 根据角色获取权限标识列表
	FindPermsByRoles(ctx context.Context, roles []string) ([]string, error)

	// FindPermsByRolesWithType 根据角色获取权限标识列表（按菜单类型过滤）
	FindPermsByRolesWithType(ctx context.Context, roles []string, menuType int) ([]string, error)

	// GetOptions 获取菜单下拉选项（扁平列表）
	GetOptions(ctx context.Context) ([]read.Option, error)

	// GetMenuOptions 获取菜单下拉选项（带树形结构）
	GetMenuOptions(ctx context.Context) ([]read.MenuOptionRead, error)

	// GetMenuRoutes 获取菜单路由列表
	GetMenuRoutes(ctx context.Context, roles []string) ([]read.MenuRouteRead, error)

	// GetFormData 获取菜单表单数据
	GetFormData(ctx context.Context, menuID int64) (*bo.MenuForm, error)

	// FindPermsByRoleCode 根据单个角色编码获取权限标识列表（用于缓存刷新）
	FindPermsByRoleCode(ctx context.Context, roleCode string) ([]string, error)

	// DeleteCascadeByIDs 批量级联删除：删除所有传入ID对应的菜单及其子孙菜单
	DeleteCascadeByIDs(ctx context.Context, ids []int64) (int64, error)

	// DeleteRoleMenuByMenuIDs 批量删除角色-菜单关联关系（含所有传入菜单及子孙菜单的关联）
	DeleteRoleMenuByMenuIDs(ctx context.Context, ids []int64) error

	// CountByIDs 统计给定ID集合中存在的菜单数量
	CountByIDs(ctx context.Context, ids []int64) (int64, error)

	// SaveRoleMenu 新增角色-菜单关联
	SaveRoleMenu(ctx context.Context, roleID, menuID int64) error

	// Transaction 执行事务
	Transaction(ctx context.Context, fn func(repo IMenuRepository) error) error
}
