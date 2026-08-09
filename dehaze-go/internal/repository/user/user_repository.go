package user

import (
	"context"
	"errors"
	"strings"
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
		InstanceSet("skip_data_scope", true).
		Where("id = ?", id).
		First(&user).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &user, err
}

// ExistsByUsername 检查用户名是否存在（含软删行）
func (r *UserRepository) ExistsByUsername(ctx context.Context, username string, excludeID ...int64) (bool, error) {
	var count int64
	query := r.db.WithContext(ctx).Unscoped().Model(&model.SysUser{}).
		Where("username = ?", username)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// ExistsByMobile 检查手机号是否存在（含软删行）
func (r *UserRepository) ExistsByMobile(ctx context.Context, mobile string, excludeID ...int64) (bool, error) {
	if mobile == "" {
		return false, nil
	}
	var count int64
	query := r.db.WithContext(ctx).Unscoped().Model(&model.SysUser{}).
		Where("mobile = ?", mobile)
	if len(excludeID) > 0 {
		query = query.Where("id != ?", excludeID[0])
	}
	err := query.Count(&count).Error
	return count > 0, err
}

// ExistsByEmail 检查邮箱是否存在（含软删行）
func (r *UserRepository) ExistsByEmail(ctx context.Context, email string, excludeID ...int64) (bool, error) {
	if email == "" {
		return false, nil
	}
	var count int64
	query := r.db.WithContext(ctx).Unscoped().Model(&model.SysUser{}).
		Where("email = ?", email)
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
		Where("su.username != 'root'")

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
		Where("su.username != 'root'").
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

// UpdateStatusWithTime 更新用户状态（带更新时间）
func (r *UserRepository) UpdateStatusWithTime(ctx context.Context, id int64, status int8, updateTime time.Time) error {
	return r.db.WithContext(ctx).Model(&model.SysUser{}).
		Where("id = ?", id).
		Updates(map[string]interface{}{
			"status":      status,
			"update_time": updateTime,
		}).Error
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
// 单次 JOIN 查询合并用户基本信息、角色编码/数据权限和权限列表
func (r *UserRepository) FindUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error) {
	type userWithRole struct {
		UserId    int64  `gorm:"column:user_id"`
		Username  string `gorm:"column:username"`
		Nickname  string `gorm:"column:nickname"`
		DeptId    int64  `gorm:"column:dept_id"`
		Password  string `gorm:"column:password"`
		Status    int8   `gorm:"column:status"`
		Code      string `gorm:"column:code"`
		DataScope int8   `gorm:"column:data_scope"`
		Perms     string `gorm:"column:perms"`
	}
	var rows []userWithRole
	err := r.db.WithContext(ctx).
		Table("sys_user u").
		Select(`u.id as user_id, u.username, u.nickname, u.dept_id, u.password, u.status,
			r.code, r.data_scope,
			(SELECT GROUP_CONCAT(DISTINCT m.perm SEPARATOR ',')
			 FROM sys_menu m
			 JOIN sys_role_menu srm ON m.id = srm.menu_id
			 JOIN sys_user_role sur2 ON srm.role_id = sur2.role_id
			 WHERE sur2.user_id = u.id AND m.deleted = 0 AND m.perm IS NOT NULL AND m.perm != '') as perms`).
		Joins("LEFT JOIN sys_user_role sur ON u.id = sur.user_id").
		Joins("LEFT JOIN sys_role r ON sur.role_id = r.id AND r.status = 1 AND r.deleted = 0").
		Where("u.username = ? AND u.deleted = 0", username).
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}

	first := rows[0]
	authInfo := model.UserAuthInfo{
		UserId:   first.UserId,
		Username: first.Username,
		Nickname: first.Nickname,
		DeptId:   first.DeptId,
		Password: first.Password,
		Status:   first.Status,
	}

	// 聚合角色编码和最小数据权限
	roles := make([]string, 0, len(rows))
	var minDataScope int8
	hasDataScope := false
	for _, row := range rows {
		if row.Code != "" {
			roles = append(roles, row.Code)
			if !hasDataScope || row.DataScope < minDataScope {
				minDataScope = row.DataScope
				hasDataScope = true
			}
		}
	}
	authInfo.Roles = roles
	if hasDataScope {
		authInfo.DataScope = minDataScope
	}

	// 解析权限列表（从 GROUP_CONCAT 结果拆分）
	if first.Perms != "" {
		authInfo.Perms = strings.Split(first.Perms, ",")
	} else {
		authInfo.Perms = []string{}
	}

	return &authInfo, nil
}

func (r *UserRepository) FindUserAuthInfoByID(ctx context.Context, userID int64) (*model.UserAuthInfo, error) {
	type userWithRole struct {
		UserId    int64  `gorm:"column:user_id"`
		Username  string `gorm:"column:username"`
		Nickname  string `gorm:"column:nickname"`
		DeptId    int64  `gorm:"column:dept_id"`
		Password  string `gorm:"column:password"`
		Status    int8   `gorm:"column:status"`
		Code      string `gorm:"column:code"`
		DataScope int8   `gorm:"column:data_scope"`
		Perms     string `gorm:"column:perms"`
	}
	var rows []userWithRole
	err := r.db.WithContext(ctx).
		Table("sys_user u").
		Select(`u.id as user_id, u.username, u.nickname, u.dept_id, u.password, u.status,
			r.code, r.data_scope,
			(SELECT GROUP_CONCAT(DISTINCT m.perm SEPARATOR ',')
			 FROM sys_menu m
			 JOIN sys_role_menu srm ON m.id = srm.menu_id
			 JOIN sys_user_role sur2 ON srm.role_id = sur2.role_id
			 WHERE sur2.user_id = u.id AND m.deleted = 0 AND m.perm IS NOT NULL AND m.perm != '') as perms`).
		Joins("LEFT JOIN sys_user_role sur ON u.id = sur.user_id").
		Joins("LEFT JOIN sys_role r ON sur.role_id = r.id AND r.status = 1 AND r.deleted = 0").
		Where("u.id = ? AND u.deleted = 0", userID).
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}

	first := rows[0]
	authInfo := model.UserAuthInfo{
		UserId:   first.UserId,
		Username: first.Username,
		Nickname: first.Nickname,
		DeptId:   first.DeptId,
		Password: first.Password,
		Status:   first.Status,
	}

	roles := make([]string, 0, len(rows))
	var minDataScope int8
	hasDataScope := false
	for _, row := range rows {
		if row.Code != "" {
			roles = append(roles, row.Code)
			if !hasDataScope || row.DataScope < minDataScope {
				minDataScope = row.DataScope
				hasDataScope = true
			}
		}
	}
	authInfo.Roles = roles
	if hasDataScope {
		authInfo.DataScope = minDataScope
	}

	if first.Perms != "" {
		authInfo.Perms = strings.Split(first.Perms, ",")
	} else {
		authInfo.Perms = []string{}
	}

	return &authInfo, nil
}

// FindUserWithRoleCodesByID 根据用户ID查询用户信息和角色编码（单次JOIN查询，消除N+1）
func (r *UserRepository) FindUserWithRoleCodesByID(ctx context.Context, userID int64) (*model.SysUser, []string, error) {
	type userWithRoleCode struct {
		ID       int64  `gorm:"column:id"`
		Username string `gorm:"column:username"`
		Nickname string `gorm:"column:nickname"`
		Avatar   string `gorm:"column:avatar"`
		Code     string `gorm:"column:code"`
	}
	var rows []userWithRoleCode
	err := r.db.WithContext(ctx).
		Table("sys_user u").
		Select("u.id, u.username, u.nickname, u.avatar, r.code").
		Joins("LEFT JOIN sys_user_role sur ON u.id = sur.user_id").
		Joins("LEFT JOIN sys_role r ON sur.role_id = r.id").
		Where("u.id = ? AND u.deleted = 0", userID).
		Scan(&rows).Error
	if err != nil {
		return nil, nil, err
	}
	if len(rows) == 0 {
		return nil, nil, nil
	}
	first := rows[0]
	user := &model.SysUser{
		BaseModel: model.BaseModel{ID: first.ID},
		Username:  first.Username,
		Nickname:  first.Nickname,
		Avatar:    first.Avatar,
	}
	roles := make([]string, 0, len(rows))
	for _, row := range rows {
		if row.Code != "" {
			roles = append(roles, row.Code)
		}
	}
	return user, roles, nil
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
		InstanceSet("skip_data_scope", true).
		Select("id, username, nickname, mobile, email, gender, avatar, dept_id, status").
		Where("id = ?", userID).
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
			Where("id = ?", userID).
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

// FindUsernameByID 查询用户名
func (r *UserRepository) FindUsernameByID(ctx context.Context, userID int64) (string, error) {
	var username string
	err := r.db.WithContext(ctx).
		Table("sys_user").
		Where("id = ? AND deleted = 0", userID).
		Select("username").
		Scan(&username).Error
	return username, err
}

// FindUsernamesByIDs 批量查询用户名
func (r *UserRepository) FindUsernamesByIDs(ctx context.Context, ids []int64) (map[int64]string, error) {
	result := make(map[int64]string)
	if len(ids) == 0 {
		return result, nil
	}
	type row struct {
		ID       int64  `gorm:"column:id"`
		Username string `gorm:"column:username"`
	}
	var rows []row
	err := r.db.WithContext(ctx).
		Table("sys_user").
		Where("id IN ? AND deleted = 0", ids).
		Select("id, username").
		Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, r := range rows {
		result[r.ID] = r.Username
	}
	return result, nil
}

// FindUserInfoByID 查询用户名和头像
func (r *UserRepository) FindUserInfoByID(ctx context.Context, userID int64) (string, string, error) {
	type userRow struct {
		Username string `gorm:"column:username"`
		Avatar   string `gorm:"column:avatar"`
	}
	var row userRow
	err := r.db.WithContext(ctx).
		Table("sys_user").
		Where("id = ? AND deleted = 0", userID).
		Select("username, avatar").
		Scan(&row).Error
	return row.Username, row.Avatar, err
}

// FindAdminUserIDs 查询所有管理员用户ID（ROOT/ADMIN 角色）
func (r *UserRepository) FindAdminUserIDs(ctx context.Context) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).
		Table("sys_user u").
		Joins("INNER JOIN sys_user_role ur ON u.id = ur.user_id").
		Joins("INNER JOIN sys_role r ON ur.role_id = r.id").
		Where("r.code IN ? AND u.deleted = 0 AND u.status = 1", []string{"ROOT", "ADMIN"}).
		Pluck("DISTINCT u.id", &ids).Error
	return ids, err
}

// FindAllActiveUserIDs 查询所有正常状态用户ID
func (r *UserRepository) FindAllActiveUserIDs(ctx context.Context) ([]int64, error) {
	var ids []int64
	err := r.db.WithContext(ctx).
		Model(&model.SysUser{}).
		Where("deleted = 0 AND status = 1").
		Pluck("id", &ids).Error
	return ids, err
}

// Ensure UserRepository implements IUserRepository
var _ IUserRepository = (*UserRepository)(nil)
