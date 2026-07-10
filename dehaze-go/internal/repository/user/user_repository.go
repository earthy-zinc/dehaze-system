package user

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"

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

// ExistsByMobile 检查手机号是否存在
func (r *UserRepository) ExistsByMobile(ctx context.Context, mobile string, excludeID ...int64) (bool, error) {
	if mobile == "" {
		return false, nil
	}
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("mobile = ? AND deleted = 0", mobile)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// ExistsByEmail 检查邮箱是否存在
func (r *UserRepository) ExistsByEmail(ctx context.Context, email string, excludeID ...int64) (bool, error) {
	if email == "" {
		return false, nil
	}
	var count int64
	query := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("email = ? AND deleted = 0", email)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// FindPage 分页查询用户列表
func (r *UserRepository) FindPage(ctx context.Context, q *query.UserPageQuery) (*read.PageResult[read.UserPage], error) {
	var users []read.UserPage
	var total int64

	// 基础计数查询（不含 GROUP BY）
	countDB := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Table("sys_user su").
		Where("su.deleted = 0 AND su.username != 'root'")

	// 主数据查询（含 GROUP BY 用于聚合角色名称）
	dataDB := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Select(`su.id, su.username, su.nickname, su.mobile, su.email, su.avatar,
                su.status, su.dept_id, sd.name as dept_name,
                CASE su.gender WHEN 1 THEN '男' WHEN 2 THEN '女' ELSE '未知' END as gender_label,
                GROUP_CONCAT(sr.name SEPARATOR ',') as role_names,
                su.create_time`).
		Table("sys_user su").
		Joins("LEFT JOIN sys_dept sd ON su.dept_id = sd.id").
		Joins("LEFT JOIN sys_user_role sur ON su.id = sur.user_id").
		Joins("LEFT JOIN sys_role sr ON sur.role_id = sr.id AND sr.deleted = 0").
		Where("su.deleted = 0 AND su.username != 'root'").
		Group("su.id")

	// 构建查询条件（两个查询共用）
	applyFilters := func(db *gorm.DB) *gorm.DB {
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
		return db
	}
	countDB = applyFilters(countDB)
	dataDB = applyFilters(dataDB)

	// 统计总数（不含 GROUP BY 的独立查询）
	if err := countDB.Count(&total).Error; err != nil {
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

	err := dataDB.Order("su.create_time DESC").
		Offset((pageNum - 1) * pageSize).
		Limit(pageSize).
		Scan(&users).Error
	if err != nil {
		return nil, err
	}

	return &read.PageResult[read.UserPage]{
		List:  users,
		Total: total,
	}, nil
}

// FindPageWithRoles 分页查询用户列表（含角色名称）
func (r *UserRepository) FindPageWithRoles(ctx context.Context, q *query.UserPageQuery, deptIDs []int64) ([]read.UserPageWithRoles, int64, error) {
	// 初始化分页参数
	pageNum := q.PageNum
	pageSize := q.PageSize
	if pageNum <= 0 {
		pageNum = 1
	}
	if pageSize <= 0 {
		pageSize = 10
	}

	queryBuilder := r.db.WithContext(ctx).Table("sys_user u").
		Select("u.id, u.username, u.nickname, u.mobile, u.gender, u.avatar, u.status, u.email, d.name as dept_name, GROUP_CONCAT(r.name) as role_names, u.create_time").
		Joins("LEFT JOIN sys_dept d ON u.dept_id = d.id").
		Joins("LEFT JOIN sys_user_role sur ON u.id = sur.user_id").
		Joins("LEFT JOIN sys_role r ON sur.role_id = r.id").
		Where("u.deleted = 0 AND u.username != 'root'").
		Group("u.id")

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		queryBuilder = queryBuilder.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", keyword, keyword, keyword)
	}
	if q.Status != nil {
		queryBuilder = queryBuilder.Where("u.status = ?", *q.Status)
	}
	if len(deptIDs) > 0 {
		queryBuilder = queryBuilder.Where("u.dept_id IN ?", deptIDs)
	} else if q.DeptId != nil {
		queryBuilder = queryBuilder.Where("u.dept_id = ?", *q.DeptId)
	}
	if q.StartTime != "" {
		queryBuilder = queryBuilder.Where("u.create_time >= ?", q.StartTime)
	}
	if q.EndTime != "" {
		queryBuilder = queryBuilder.Where("u.create_time <= ?", q.EndTime)
	}

	var total int64
	if err := queryBuilder.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	var results []read.UserPageWithRoles
	err := queryBuilder.Offset((pageNum - 1) * pageSize).Limit(pageSize).Find(&results).Error
	if err != nil {
		return nil, 0, err
	}

	return results, total, nil
}

// FindExportUsers 导出用户列表
func (r *UserRepository) FindExportUsers(ctx context.Context, q *query.UserPageQuery) ([]read.UserExport, error) {
	queryBuilder := r.db.WithContext(ctx).Table("sys_user u").
		Select("u.username, u.nickname, u.mobile, u.email, u.status, " +
			"CASE u.gender WHEN 1 THEN '男' WHEN 2 THEN '女' ELSE '未知' END as gender, " +
			"d.name as dept_name, u.create_time").
		Joins("LEFT JOIN sys_dept d ON u.dept_id = d.id").
		Where("u.deleted = 0 AND u.username != 'root'")

	if q.Keywords != "" {
		keyword := "%" + q.Keywords + "%"
		queryBuilder = queryBuilder.Where("u.username LIKE ? OR u.nickname LIKE ? OR u.mobile LIKE ?", keyword, keyword, keyword)
	}
	if q.Status != nil {
		queryBuilder = queryBuilder.Where("u.status = ?", *q.Status)
	}
	if q.DeptId != nil {
		queryBuilder = queryBuilder.Where("u.dept_id = ?", *q.DeptId)
	}

	var exportData []struct {
		Username   string    `json:"username"`
		Nickname   string    `json:"nickname"`
		DeptName   string    `json:"dept_name"`
		Gender     string    `json:"gender"`
		Mobile     string    `json:"mobile"`
		Email      string    `json:"email"`
		Status     int8      `json:"status"`
		CreateTime time.Time `json:"create_time"`
	}

	if err := queryBuilder.Find(&exportData).Error; err != nil {
		return nil, err
	}

	userExports := make([]read.UserExport, 0, len(exportData))
	for _, data := range exportData {
		statusLabel := "禁用"
		if data.Status == 1 {
			statusLabel = "启用"
		}
		userExports = append(userExports, read.UserExport{
			Username:    data.Username,
			Nickname:    data.Nickname,
			DeptName:    data.DeptName,
			Gender:      data.Gender,
			Mobile:      data.Mobile,
			Email:       data.Email,
			StatusLabel: statusLabel,
			CreateTime:  data.CreateTime,
		})
	}

	return userExports, nil
}

// FindRoleCodesByUsername 根据用户名查询角色编码列表
func (r *UserRepository) FindRoleCodesByUsername(ctx context.Context, username string) ([]string, error) {
	var roles []string
	err := r.db.WithContext(ctx).Table("sys_user u").
		Select("r.code").
		Joins("LEFT JOIN sys_user_role ur ON u.id = ur.user_id").
		Joins("LEFT JOIN sys_role r ON ur.role_id = r.id").
		Where("u.username = ? AND u.deleted = 0 AND r.code IS NOT NULL", username).
		Pluck("r.code", &roles).Error
	return roles, err
}

// ExistsRootInIDs 检查是否包含超级管理员
func (r *UserRepository) ExistsRootInIDs(ctx context.Context, ids []int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id IN ? AND username = ?", ids, "root").
		Count(&count).Error
	return count > 0, err
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

// UpdateStatusWithTime 更新用户状态（带更新时间）
func (r *UserRepository) UpdateStatusWithTime(ctx context.Context, id int64, status int8, updateTime time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"status":      status,
			"update_time": updateTime,
		}).Error
}

// UpdatePassword 更新用户密码
func (r *UserRepository) UpdatePassword(ctx context.Context, id int64, password string) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Update("password", password).Error
}

// UpdatePasswordWithTime 更新用户密码（带更新时间）
func (r *UserRepository) UpdatePasswordWithTime(ctx context.Context, id int64, password string, updateTime time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"password":    password,
			"update_time": updateTime,
		}).Error
}

// Delete 删除用户（逻辑删除）
func (r *UserRepository) Delete(ctx context.Context, ids []int64) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id IN ?", ids).
		Update("deleted", 1).Error
}

// SoftDeleteWithTime 逻辑删除用户（带更新时间）
func (r *UserRepository) SoftDeleteWithTime(ctx context.Context, ids []int64, updateTime time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id IN ?", ids).
		Updates(map[string]interface{}{
			"deleted":     1,
			"update_time": updateTime,
		}).Error
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
	// 删除原有角色
	if err := r.db.WithContext(ctx).Where("user_id = ?", userID).Delete(&model.SysUserRole{}).Error; err != nil {
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
		return r.db.WithContext(ctx).Create(&userRoles).Error
	}
	return nil
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

// Transaction 事务执行（在同一事务中完成多个仓储操作）
func (r *UserRepository) Transaction(ctx context.Context, fn func(repo IUserRepository) error) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		return fn(&UserRepository{db: tx})
	})
}

// CreateWithRoles 创建用户并分配角色（事务）
func (r *UserRepository) CreateWithRoles(ctx context.Context, user *model.SysUser, roleIDs []int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 创建用户
		if err := tx.Create(user).Error; err != nil {
			return err
		}

		// 分配角色
		if len(roleIDs) > 0 {
			userRoles := make([]model.SysUserRole, 0, len(roleIDs))
			for _, roleID := range roleIDs {
				userRoles = append(userRoles, model.SysUserRole{
					UserID: user.ID,
					RoleID: roleID,
				})
			}
			if err := tx.Create(&userRoles).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// UpdateWithRoles 更新用户并更新角色（事务）
func (r *UserRepository) UpdateWithRoles(ctx context.Context, userID int64, updates map[string]interface{}, roleIDs []int64) error {
	return r.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		// 更新用户
		if err := tx.Model(&model.SysUser{}).
			Where("id = ? AND deleted = 0", userID).
			Updates(updates).Error; err != nil {
			return err
		}

		// 删除原有角色关联
		if err := tx.Where("user_id = ?", userID).Delete(&model.SysUserRole{}).Error; err != nil {
			return err
		}

		// 添加新角色关联
		if len(roleIDs) > 0 {
			userRoles := make([]model.SysUserRole, 0, len(roleIDs))
			for _, roleID := range roleIDs {
				userRoles = append(userRoles, model.SysUserRole{
					UserID: userID,
					RoleID: roleID,
				})
			}
			if err := tx.Create(&userRoles).Error; err != nil {
				return err
			}
		}
		return nil
	})
}

// ImportUserInTx 在事务中导入用户（由 Transaction 回调内调用）
func (r *UserRepository) ImportUserInTx(ctx context.Context, user *model.SysUser) error {
	return r.db.WithContext(ctx).Create(user).Error
}

// ExistsByUsernameInTx 在事务中检查用户名是否存在（由 Transaction 回调内调用）
func (r *UserRepository) ExistsByUsernameInTx(ctx context.Context, username string) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("username = ? AND deleted = 0", username).
		Count(&count).Error
	return count > 0, err
}

// Ensure UserRepository implements IUserRepository
var _ IUserRepository = (*UserRepository)(nil)
