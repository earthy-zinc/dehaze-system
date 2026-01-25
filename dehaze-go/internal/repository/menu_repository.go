package repository

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"

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

// GetOptions 获取菜单下拉选项
func (r *MenuRepository) GetOptions(ctx context.Context) ([]vo.Option, error) {
	var options []vo.Option
	err := r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("id as value, name as label").
		Where("type IN (1, 2)").
		Order("sort ASC").
		Scan(&options).Error
	return options, err
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

// Ensure MenuRepository implements IMenuRepository
var _ IMenuRepository = (*MenuRepository)(nil)
