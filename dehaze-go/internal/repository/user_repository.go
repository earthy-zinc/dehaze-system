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

// UserRepository 用户仓储实现
type UserRepository struct {
	db *gorm.DB
}

// NewUserRepository 创建用户仓储实例
func NewUserRepository(db *gorm.DB) *UserRepository {
	return &UserRepository{db: db}
}

// FindByID 根据 ID 查询用户
func (r *UserRepository) FindByID(ctx context.Context, id int64) (*model.SysUser, error) {
	var user model.SysUser
	err := r.db.WithContext(ctx).
		Where("id = ? AND deleted = 0", id).
		First(&user).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &user, err
}

// FindByUsername 根据用户名查询用户
func (r *UserRepository) FindByUsername(ctx context.Context, username string) (*model.SysUser, error) {
	var user model.SysUser
	err := r.db.WithContext(ctx).
		Where("username = ? AND deleted = 0", username).
		First(&user).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &user, err
}

// ExistsByUsername 检查用户名是否存在
func (r *UserRepository) ExistsByUsername(ctx context.Context, username string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("username = ? AND deleted = 0", username)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// FindPage 分页查询用户列表
func (r *UserRepository) FindPage(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error) {
	var users []vo.UserPageVO
	var total int64

	db := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Select(`su.id, su.username, su.nickname, su.mobile, su.gender, su.avatar,
                su.status, su.dept_id, sd.name as dept_name, su.create_time`).
		Table("sys_user su").
		Joins("LEFT JOIN sys_dept sd ON su.dept_id = sd.id").
		Where("su.deleted = 0")

	// 构建查询条件
	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		db = db.Where("(su.username LIKE ? OR su.nickname LIKE ? OR su.mobile LIKE ?)",
			keyword, keyword, keyword)
	}
	if q.Status != nil {
		db = db.Where("su.status = ?", *q.Status)
	}
	if q.DeptId != nil {
		db = db.Where("su.dept_id = ?", *q.DeptId)
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

	err := db.Order("su.create_time DESC").
		Offset((pageNum - 1) * pageSize).
		Limit(pageSize).
		Scan(&users).Error
	if err != nil {
		return nil, err
	}

	return &vo.PageResult[vo.UserPageVO]{
		List:  users,
		Total: total,
	}, nil
}

// Create 创建用户
func (r *UserRepository) Create(ctx context.Context, user *model.SysUser) error {
	return r.db.WithContext(ctx).Create(user).Error
}

// Update 更新用户
func (r *UserRepository) Update(ctx context.Context, user *model.SysUser) error {
	return r.db.WithContext(ctx).Model(user).
		Select("nickname", "mobile", "email", "gender", "avatar", "dept_id", "status", "update_by").
		Updates(user).Error
}

// UpdateStatus 更新用户状态
func (r *UserRepository) UpdateStatus(ctx context.Context, id int64, status int8) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Update("status", status).Error
}

// UpdatePassword 更新用户密码
func (r *UserRepository) UpdatePassword(ctx context.Context, id int64, password string) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Update("password", password).Error
}

// Delete 删除用户（逻辑删除）
func (r *UserRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id IN ?", ids).
		Update("deleted", 1).Error
}

// FindUserAuthInfo 查询用户认证信息（含角色、权限）
func (r *UserRepository) FindUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error) {
	var authInfo model.UserAuthInfo

	// 查询用户基本信息
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Select("id as user_id, username, nickname, dept_id, password, status").
		Where("username = ? AND deleted = 0", username).
		Scan(&authInfo).Error
	if err != nil {
		return nil, err
	}
	if authInfo.UserId == 0 {
		return nil, nil
	}

	// 查询用户角色
	var roles []string
	err = r.db.WithContext(ctx).
		Model(&model.SysRole{}).
		Select("code").
		Joins("JOIN sys_user_role sur ON sys_role.id = sur.role_id").
		Where("sur.user_id = ? AND sys_role.status = 1 AND sys_role.deleted = 0", authInfo.UserId).
		Scan(&roles).Error
	if err != nil {
		return nil, err
	}
	authInfo.Roles = roles

	// 查询用户权限
	var perms []string
	err = r.db.WithContext(ctx).
		Model(&model.SysMenu{}).
		Select("DISTINCT perm").
		Joins("JOIN sys_role_menu srm ON sys_menu.id = srm.menu_id").
		Joins("JOIN sys_user_role sur ON srm.role_id = sur.role_id").
		Where("sur.user_id = ? AND sys_menu.perm IS NOT NULL AND sys_menu.perm != ''", authInfo.UserId).
		Scan(&perms).Error
	if err != nil {
		return nil, err
	}
	authInfo.Perms = perms

	// 查询数据权限（取最大权限）
	var dataScope int8
	err = r.db.WithContext(ctx).
		Model(&model.SysRole{}).
		Select("MIN(data_scope)").
		Joins("JOIN sys_user_role sur ON sys_role.id = sur.role_id").
		Where("sur.user_id = ? AND sys_role.status = 1", authInfo.UserId).
		Scan(&dataScope).Error
	if err != nil {
		return nil, err
	}
	authInfo.DataScope = dataScope

	return &authInfo, nil
}

// AssignRoles 分配用户角色
func (r *UserRepository) AssignRoles(ctx context.Context, userID int64, roleIDs []int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 删除原有角色
		if err := tx.Where("user_id = ?", userID).Delete(&model.SysUserRole{}).Error; err != nil {
			return err
		}
		// 添加新角色
		if len(roleIDs) > 0 {
			userRoles := make([]model.SysUserRole, 0, len(roleIDs))
			for _, roleID := range roleIDs {
				userRoles = append(userRoles, model.SysUserRole{
					UserID: userID,
					RoleID: roleID,
				})
			}
			return tx.Create(&userRoles).Error
		}
		return nil
	})
}

// GetUserRoleIDs 获取用户角色 ID 列表
func (r *UserRepository) GetUserRoleIDs(ctx context.Context, userID int64) ([]int64, error) {
	var roleIDs []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUserRole{}).
		Select("role_id").
		Where("user_id = ?", userID).
		Scan(&roleIDs).Error
	return roleIDs, err
}

// GetFormData 获取用户表单数据
func (r *UserRepository) GetFormData(ctx context.Context, userID int64) (*bo.UserFormBO, error) {
	var form bo.UserFormBO
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Select("id, username, nickname, mobile, email, gender, avatar, dept_id, status").
		Where("id = ? AND deleted = 0", userID).
		Scan(&form).Error
	if err != nil {
		return nil, err
	}
	if form.ID == 0 {
		return nil, nil
	}

	// 获取角色 ID 列表
	roleIDs, err := r.GetUserRoleIDs(ctx, userID)
	if err != nil {
		return nil, err
	}
	form.RoleIds = roleIDs

	return &form, nil
}

// Ensure UserRepository implements IUserRepository
var _ IUserRepository = (*UserRepository)(nil)
