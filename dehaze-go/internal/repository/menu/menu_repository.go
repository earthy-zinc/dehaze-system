package menu

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"

	"gorm.io/gorm"
)

// MenuRepository 菜单仓储实现
type MenuRepository struct {
	db *gorm.DB
}

// NewMenuRepository 创建菜单仓储实例
func NewMenuRepository(db *gorm.DB) *MenuRepository {
	return &MenuRepository{db: db}
}

// FindByID 根据 ID 查询菜单
func (r *MenuRepository) FindByID(ctx context.Context, id int64) (*model.SysMenu, error) {
	var menu model.SysMenu
	err := r.db.WithContext(ctx).
		Where("id = ?", id).
		First(&menu).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &menu, err
}

// FindAll 查询所有菜单
func (r *MenuRepository) FindAll(ctx context.Context, q *query.MenuQuery) ([]model.SysMenu, error) {
	var menus []model.SysMenu
	db := r.db.WithContext(ctx).Model(&model.SysMenu{})

	if q != nil {
		if q.Keywords != "" {
			keyword := "%" + q.Keywords + "%"
			db = db.Where("name LIKE ?", keyword)
		}
		if q.Status != nil {
			db = db.Where("visible = ?", *q.Status)
		}
	}

	err := db.Order("sort ASC").Find(&menus).Error
	return menus, err
}

// FindByParentID 根据父 ID 查询子菜单
func (r *MenuRepository) FindByParentID(ctx context.Context, parentID int64) ([]model.SysMenu, error) {
	var menus []model.SysMenu
	err := r.db.WithContext(ctx).
		Where("parent_id = ?", parentID).
		Order("sort ASC").
		Find(&menus).Error
	return menus, err
}

// Create 创建菜单
func (r *MenuRepository) Create(ctx context.Context, menu *model.SysMenu) error {
	return r.db.WithContext(ctx).Create(menu).Error
}

// Update 更新菜单
func (r *MenuRepository) Update(ctx context.Context, menu *model.SysMenu) error {
	return r.db.WithContext(ctx).Model(menu).
		Select("parent_id", "tree_path", "name", "type", "path", "component",
			"perm", "visible", "sort", "icon", "redirect", "always_show", "keep_alive").
		Updates(menu).Error
}

// Delete 删除菜单
func (r *MenuRepository) Delete(ctx context.Context, id int64) error {
	return r.db.WithContext(ctx).Delete(&model.SysMenu{}, id).Error
}

// HasChildren 检查菜单是否有子菜单
func (r *MenuRepository) HasChildren(ctx context.Context, id int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Where("parent_id = ?", id).
		Count(&count).Error
	return count > 0, err
}

// FindRoutesByRoles 根据角色获取路由菜单
func (r *MenuRepository) FindRoutesByRoles(ctx context.Context, roles []string) ([]model.SysMenu, error) {
	var menus []model.SysMenu

	// 如果是超级管理员（ROOT），返回所有菜单
	isRoot := false
	for _, role := range roles {
		if role == "ROOT" {
			isRoot = true
			break
		}
	}

	if isRoot {
		err := r.db.WithContext(ctx).
			Where("type IN (1, 2) AND visible = 1").
			Order("sort ASC").
			Find(&menus).Error
		return menus, err
	}

	// 普通用户根据角色查询菜单
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Distinct("sys_menu.*").
		Joins("JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
		Joins("JOIN sys_role sr ON srm.role_id = sr.id").
		Where("sr.code IN ? AND sr.status = 1 AND sr.deleted = 0", roles).
		Where("sys_menu.type IN (1, 2) AND sys_menu.visible = 1").
		Order("sys_menu.sort ASC").
		Find(&menus).Error
	return menus, err
}

// FindPermsByRoles 根据角色获取权限标识列表
func (r *MenuRepository) FindPermsByRoles(ctx context.Context, roles []string) ([]string, error) {
	var perms []string

	// 如果是超级管理员（ROOT），返回所有权限
	isRoot := false
	for _, role := range roles {
		if role == "ROOT" {
			isRoot = true
			break
		}
	}

	if isRoot {
		err := r.db.WithContext(ctx).
			Model(&model.SysMenu{}).
			Select("DISTINCT perm").
			Where("perm IS NOT NULL AND perm != ''").
			Scan(&perms).Error
		return perms, err
	}

	// 普通用户根据角色查询权限
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("DISTINCT sys_menu.perm").
		Joins("JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
		Joins("JOIN sys_role sr ON srm.role_id = sr.id").
		Where("sr.code IN ? AND sr.status = 1 AND sr.deleted = 0", roles).
		Where("sys_menu.perm IS NOT NULL AND sys_menu.perm != ''").
		Scan(&perms).Error
	return perms, err
}

// FindPermsByRolesWithType 根据角色获取权限标识列表（按菜单类型过滤）
func (r *MenuRepository) FindPermsByRolesWithType(ctx context.Context, roles []string, menuType int) ([]string, error) {
	var perms []string

	isRoot := false
	for _, role := range roles {
		if role == "ROOT" {
			isRoot = true
			break
		}
	}

	if isRoot {
		err := r.db.WithContext(ctx).
			Model(&model.SysMenu{}).
			Select("DISTINCT perm").
			Where("type = ?", menuType).
			Where("perm IS NOT NULL AND perm != ''").
			Scan(&perms).Error
		return perms, err
	}

	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("DISTINCT sys_menu.perm").
		Joins("JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
		Joins("JOIN sys_role sr ON srm.role_id = sr.id").
		Where("sr.code IN ? AND sr.status = 1 AND sr.deleted = 0", roles).
		Where("sys_menu.type = ?", menuType).
		Where("sys_menu.perm IS NOT NULL AND sys_menu.perm != ''").
		Scan(&perms).Error
	return perms, err
}

// GetOptions 获取菜单下拉选项（扁平列表）
func (r *MenuRepository) GetOptions(ctx context.Context) ([]read.Option, error) {
	var options []read.Option
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("id as value, name as label").
		Where("type IN (1, 2)").
		Order("sort ASC").
		Scan(&options).Error
	return options, err
}

// GetMenuOptions 获取菜单下拉选项（带树形结构）
func (r *MenuRepository) GetMenuOptions(ctx context.Context) ([]read.MenuOptionRead, error) {
	var options []read.MenuOptionRead
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("id, parent_id, name, type, sort").
		Where("type IN (1, 2)").
		Order("sort ASC").
		Scan(&options).Error
	return options, err
}

// GetMenuRoutes 获取菜单路由列表
func (r *MenuRepository) GetMenuRoutes(ctx context.Context, roles []string) ([]read.MenuRouteRead, error) {
	var routes []read.MenuRouteRead

	// 如果是超级管理员（ROOT），返回所有菜单
	isRoot := false
	for _, role := range roles {
		if role == "ROOT" {
			isRoot = true
			break
		}
	}

	if isRoot {
		err := r.db.WithContext(ctx).
			Model(&model.SysMenu{}).
			Select("sys_menu.id, sys_menu.parent_id, sys_menu.name, sys_menu.type, sys_menu.path, sys_menu.component, sys_menu.perm, sys_menu.visible, sys_menu.sort, sys_menu.icon, sys_menu.redirect, sys_menu.always_show, sys_menu.keep_alive, GROUP_CONCAT(DISTINCT sr.code) as roles").
			Joins("LEFT JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
			Joins("LEFT JOIN sys_role sr ON srm.role_id = sr.id AND sr.status = 1 AND sr.deleted = 0").
			Where("sys_menu.type IN (1, 2) AND sys_menu.visible = 1").
			Group("sys_menu.id").
			Order("sys_menu.sort ASC").
			Scan(&routes).Error
		return routes, err
	}

	// 普通用户根据角色查询菜单
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("sys_menu.id, sys_menu.parent_id, sys_menu.name, sys_menu.type, sys_menu.path, sys_menu.component, sys_menu.perm, sys_menu.visible, sys_menu.sort, sys_menu.icon, sys_menu.redirect, sys_menu.always_show, sys_menu.keep_alive, GROUP_CONCAT(DISTINCT sr2.code) as roles").
		Joins("JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
		Joins("JOIN sys_role sr ON srm.role_id = sr.id").
		Joins("LEFT JOIN sys_role_menu srm2 ON sys_menu.id = srm2.menu_id").
		Joins("LEFT JOIN sys_role sr2 ON srm2.role_id = sr2.id AND sr2.status = 1 AND sr2.deleted = 0").
		Where("sr.code IN ? AND sr.status = 1 AND sr.deleted = 0", roles).
		Where("sys_menu.type IN (1, 2) AND sys_menu.visible = 1").
		Group("sys_menu.id").
		Order("sys_menu.sort ASC").
		Scan(&routes).Error
	return routes, err
}

// GetFormData 获取菜单表单数据
func (r *MenuRepository) GetFormData(ctx context.Context, menuID int64) (*bo.MenuForm, error) {
	var form bo.MenuForm
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("id, parent_id, name, type, path, component, perm, visible, sort, icon, redirect, always_show, keep_alive").
		Where("id = ?", menuID).
		Scan(&form).Error
	if err != nil {
		return nil, err
	}
	if form.ID == nil {
		return nil, nil
	}
	return &form, nil
}

// FindPermsByRoleCode 根据单个角色编码获取权限标识列表（用于缓存刷新）
func (r *MenuRepository) FindPermsByRoleCode(ctx context.Context, roleCode string) ([]string, error) {
	var perms []string
	err := r.db.WithContext(ctx).
		Table("sys_menu").
		Select("DISTINCT sys_menu.perm").
		Joins("INNER JOIN sys_role_menu ON sys_menu.id = sys_role_menu.menu_id").
		Joins("INNER JOIN sys_role ON sys_role.id = sys_role_menu.role_id").
		Where("sys_role.code = ? AND sys_menu.perm IS NOT NULL AND sys_menu.perm != ''", roleCode).
		Pluck("perm", &perms).Error
	return perms, err
}

// ExistsByName 检查同级菜单名称是否存在
func (r *MenuRepository) ExistsByName(ctx context.Context, parentID int64, name string, excludeID int64) (bool, error) {
	var count int64
	db := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Where("parent_id = ? AND name = ?", parentID, name)
	if excludeID > 0 {
		db = db.Where("id != ?", excludeID)
	}
	err := db.Count(&count).Error
	return count > 0, err
}

// ExistsByPath 检查同级菜单路径是否存在
func (r *MenuRepository) ExistsByPath(ctx context.Context, parentID int64, path string, excludeID int64) (bool, error) {
	if path == "" {
		return false, nil
	}
	var count int64
	db := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Where("parent_id = ? AND path = ?", parentID, path)
	if excludeID > 0 {
		db = db.Where("id != ?", excludeID)
	}
	err := db.Count(&count).Error
	return count > 0, err
}

// ExistsByPerm 检查权限标识是否存在（全局唯一）
func (r *MenuRepository) ExistsByPerm(ctx context.Context, perm string, excludeID int64) (bool, error) {
	if perm == "" {
		return false, nil
	}
	var count int64
	db := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Where("perm = ?", perm)
	if excludeID > 0 {
		db = db.Where("id != ?", excludeID)
	}
	err := db.Count(&count).Error
	return count > 0, err
}

// Ensure MenuRepository implements IMenuRepository
var _ IMenuRepository = (*MenuRepository)(nil)

// DeleteCascade 级联删除菜单及其所有子孙菜单
// 使用treePath字段查询所有子孙节点并批量删除
func (r *MenuRepository) DeleteCascade(ctx context.Context, id int64) (int64, error) {
	// 使用treePath级联删除：删除当前菜单及所有子孙菜单
	// treePath格式为 "0,1,2"，通过CONCAT和LIKE查询所有包含当前ID的记录
	// 参数化查询防止SQL注入
	result := r.db.WithContext(ctx).
		Where("id = ? OR CONCAT(',', tree_path, ',') LIKE CONCAT('%,', ?, ',%')", id, id).
		Delete(&model.SysMenu{})
	return result.RowsAffected, result.Error
}

// DeleteRoleMenuByMenuID 删除角色-菜单关联关系
func (r *MenuRepository) DeleteRoleMenuByMenuID(ctx context.Context, menuID int64) error {
	return r.db.WithContext(ctx).
		Exec("DELETE FROM sys_role_menu WHERE menu_id = ?", menuID).
		Error
}

// Transaction 执行事务
func (r *MenuRepository) Transaction(ctx context.Context, fn func(repo IMenuRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return fn(&MenuRepository{db: tx})
	})
}
