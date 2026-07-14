package role

import (
	"context"
	"errors"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"

	"gorm.io/gorm"
)

// ROOT_ROLE_CODE 超级管理员角色编码
const ROOT_ROLE_CODE = "ROOT"

// RoleRepository 角色仓储实现
type RoleRepository struct {
	db *gorm.DB
}

// NewRoleRepository 创建角色仓储实例
func NewRoleRepository(db *gorm.DB) *RoleRepository {
	return &RoleRepository{db: db}
}

// FindByID 根据 ID 查询角色
func (r *RoleRepository) FindByID(ctx context.Context, id int64) (*model.SysRole, error) {
	var role model.SysRole
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&role).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &role, err
}

// FindByIDs 根据 ID 列表批量查询角色
func (r *RoleRepository) FindByIDs(ctx context.Context, ids []int64) ([]*model.SysRole, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	var roles []*model.SysRole
	err := r.db.WithContext(ctx).
		Where("id IN ? AND deleted = 0", ids).
		Find(&roles).Error
	return roles, err
}

// FindByCode 根据编码查询角色
func (r *RoleRepository) FindByCode(ctx context.Context, code string) (*model.SysRole, error) {
	var role model.SysRole
	err := r.db.WithContext(ctx).
		Where("code = ? AND deleted = 0", code).
		First(&role).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &role, err
}

// ExistsByCode 检查角色编码是否存在
func (r *RoleRepository) ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysRole{}).
		Where("code = ? AND deleted = 0", code)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// ExistsByName 检查角色名称是否存在
func (r *RoleRepository) ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysRole{}).
		Where("name = ? AND deleted = 0", name)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// FindPage 分页查询角色列表
func (r *RoleRepository) FindPage(ctx context.Context, q *query.RolePageQuery) (*read.PageResult[read.RolePage], error) {
	var roles []read.RolePage
	var total int64

	db := r.db.WithContext(ctx).Model(&model.SysRole{}).
		Select("id, name, code, sort, status, data_scope, create_time").
		Where("deleted = 0")

	// 构建查询条件
	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("(name LIKE ? OR code LIKE ?)", keyword, keyword)
	}

	// 统计总数
	if err := db.Count(&total).Error; err != nil {
		return nil, err
	}

	// 分页查询
	pageNum := q.PageNum
	if pageNum < 1 {
		pageNum = 1
	}
	pageSize := q.PageSize
	if pageSize < 1 {
		pageSize = 10
	}

	err := db.Order("sort ASC, create_time DESC").
		Offset((pageNum - 1) * pageSize).
		Limit(pageSize).
		Scan(&roles).Error
	if err != nil {
		return nil, err
	}

	return &read.PageResult[read.RolePage]{
		List:     roles,
		Total:    total,
		PageNum:  pageNum,
		PageSize: pageSize,
	}, nil
}

// FindOptions 获取角色下拉选项（isRoot 为 false 时排除 ROOT 角色）
func (r *RoleRepository) FindOptions(ctx context.Context, isRoot bool) ([]read.Option, error) {
	var options []read.Option
	query := r.db.WithContext(ctx).
		Model(&model.SysRole{}).
		Select("id as value, name as label").
		Where("status = 1 AND deleted = 0").
		Order("sort ASC")

	// 非超级管理员不显示超级管理员角色
	if !isRoot {
		query = query.Where("code != ?", ROOT_ROLE_CODE)
	}

	err := query.Scan(&options).Error
	return options, err
}

// Create 创建角色
func (r *RoleRepository) Create(ctx context.Context, role *model.SysRole) error {
	return r.db.WithContext(ctx).Create(role).Error
}

// Update 更新角色
func (r *RoleRepository) Update(ctx context.Context, role *model.SysRole) error {
	return r.db.WithContext(ctx).Model(role).
		Select("name", "code", "sort", "status", "data_scope").
		Updates(role).Error
}

// UpdateStatus 更新角色状态
func (r *RoleRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).Model(&model.SysRole{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{"status": status}).Error
}

// Delete 删除角色（逻辑删除）
func (r *RoleRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Model(&model.SysRole{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{"deleted": 1}).Error
}

// HasUsers 检查角色是否关联用户
func (r *RoleRepository) HasUsers(ctx context.Context, roleID int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserRole{}).
		Where("role_id = ?", roleID).
		Count(&count).Error
	return count > 0, err
}

// HasUsersInBatch 批量检查角色是否关联用户
func (r *RoleRepository) HasUsersInBatch(ctx context.Context, roleIDs []int64) (map[int64]bool, error) {
	result := make(map[int64]bool, len(roleIDs))
	if len(roleIDs) == 0 {
		return result, nil
	}

	type roleCount struct {
		RoleID int64 `gorm:"column:role_id"`
		Count  int64 `gorm:"column:cnt"`
	}
	var counts []roleCount
	err := r.db.WithContext(ctx).
		Model(&model.SysUserRole{}).
		Select("role_id, COUNT(*) as cnt").
		Where("role_id IN ?", roleIDs).
		Group("role_id").
		Scan(&counts).Error
	if err != nil {
		return nil, err
	}

	for _, c := range counts {
		if c.Count > 0 {
			result[c.RoleID] = true
		}
	}
	return result, nil
}

// GetMenuIDs 获取角色菜单 ID 列表
func (r *RoleRepository) GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error) {
	var menuIDs []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysRoleMenu{}).
		Select("menu_id").
		Where("role_id = ?", roleID).
		Scan(&menuIDs).Error
	return menuIDs, err
}

// AssignMenus 分配角色菜单
func (r *RoleRepository) AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 删除原有菜单
		if err := tx.Where("role_id = ?", roleID).Delete(&model.SysRoleMenu{}).Error; err != nil {
			return err
		}
		// 添加新菜单
		if len(menuIDs) > 0 {
			roleMenus := make([]model.SysRoleMenu, 0, len(menuIDs))
			for _, menuID := range menuIDs {
				roleMenus = append(roleMenus, model.SysRoleMenu{
					RoleID: roleID,
					MenuID: menuID,
				})
			}
			return tx.Create(&roleMenus).Error
		}
		return nil
	})
}

// DeleteMenusByRoleIDs 批量删除角色的菜单关联
func (r *RoleRepository) DeleteMenusByRoleIDs(ctx context.Context, roleIDs []int64) error {
	if len(roleIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).
		Where("role_id IN ?", roleIDs).
		Delete(&model.SysRoleMenu{}).Error
}

// DeleteWithMenus 删除角色及其菜单关联（事务）
func (r *RoleRepository) DeleteWithMenus(ctx context.Context, roleIDs []int64) error {
	if len(roleIDs) == 0 {
		return nil
	}
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 删除角色菜单关联
		if err := tx.Where("role_id IN ?", roleIDs).Delete(&model.SysRoleMenu{}).Error; err != nil {
			return err
		}
		// 逻辑删除角色
		if err := tx.Model(&model.SysRole{}).
			Where("id IN ?", roleIDs).
			Updates(map[string]interface{}{"deleted": 1}).Error; err != nil {
			return err
		}
		return nil
	})
}

// GetFormData 获取角色表单数据
func (r *RoleRepository) GetFormData(ctx context.Context, roleID int64) (*read.RoleForm, error) {
	var form read.RoleForm
	err := r.db.WithContext(ctx).
		Model(&model.SysRole{}).
		Select("id, name, code, sort, status, data_scope").
		Where("id = ? AND deleted = 0", roleID).
		Scan(&form).Error
	if err != nil {
		return nil, err
	}
	if form.ID == nil {
		return nil, nil
	}
	return &form, nil
}

// GetMinimumDataScope 获取角色的最小数据权限范围
func (r *RoleRepository) GetMinimumDataScope(ctx context.Context, roleCodes []string) (*int8, error) {
	if len(roleCodes) == 0 {
		return nil, nil
	}
	var scope int8
	err := r.db.WithContext(ctx).
		Model(&model.SysRole{}).
		Select("MIN(data_scope)").
		Where("code IN ?", roleCodes).
		Where("deleted = 0").
		Scan(&scope).Error
	if err != nil {
		return nil, err
	}
	return &scope, nil
}

// Ensure RoleRepository implements IRoleRepository
var _ IRoleRepository = (*RoleRepository)(nil)
