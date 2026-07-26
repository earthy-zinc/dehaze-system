package user

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/read"
)

// IUserRepository 用户仓储接口
type IUserRepository interface {
	// FindByID 根据 ID 查询用户
	FindByID(ctx context.Context, id int64) (*model.SysUser, error)

	// ExistsByUsername 检查用户名是否存在
	ExistsByUsername(ctx context.Context, username string, excludeID ...int64) (bool, error)

	// ExistsByMobile 检查手机号是否存在
	ExistsByMobile(ctx context.Context, mobile string, excludeID ...int64) (bool, error)

	// ExistsByEmail 检查邮箱是否存在
	ExistsByEmail(ctx context.Context, email string, excludeID ...int64) (bool, error)

	// FindPage 分页查询用户列表
	FindPage(ctx context.Context, q *query.UserPageQuery) (*read.PageResult[read.UserPage], error)

	// FindPageWithRoles 分页查询用户列表（含角色名称）
	FindPageWithRoles(ctx context.Context, q *query.UserPageQuery, deptIDs []int64) ([]read.UserPageWithRoles, int64, error)

	// ExistsRootInIDs 检查是否包含超级管理员
	ExistsRootInIDs(ctx context.Context, ids []int64) (bool, error)

	// Create 创建用户
	Create(ctx context.Context, user *model.SysUser) error

	// UpdateStatusWithTime 更新用户状态（带更新时间）
	UpdateStatusWithTime(ctx context.Context, id int64, status int8, updateTime time.Time) error

	// UpdatePasswordWithTime 更新用户密码（带更新时间）
	UpdatePasswordWithTime(ctx context.Context, id int64, password string, updateTime time.Time) error

	// SoftDeleteWithTime 逻辑删除用户（带更新时间）
	SoftDeleteWithTime(ctx context.Context, ids []int64, updateTime time.Time) error

	// FindUserAuthInfo 查询用户认证信息（含角色、权限）
	FindUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error)

	// FindUserAuthInfoByID 根据用户ID查询认证信息（含角色、权限）
	FindUserAuthInfoByID(ctx context.Context, userID int64) (*model.UserAuthInfo, error)

	// FindUserWithRoleCodesByID 根据用户ID查询用户信息和角色编码（单次JOIN查询，消除N+1）
	FindUserWithRoleCodesByID(ctx context.Context, userID int64) (*model.SysUser, []string, error)

	// AssignRoles 分配用户角色
	AssignRoles(ctx context.Context, userID int64, roleIDs []int64) error

	// GetUserRoleIDs 获取用户角色 ID 列表
	GetUserRoleIDs(ctx context.Context, userID int64) ([]int64, error)

	// GetFormData 获取用户表单数据
	GetFormData(ctx context.Context, userID int64) (*bo.UserFormBO, error)

	// Transaction 事务执行（在同一事务中完成多个仓储操作）
	Transaction(ctx context.Context, fn func(repo IUserRepository) error) error

	// CreateWithRoles 创建用户并分配角色（事务）
	CreateWithRoles(ctx context.Context, user *model.SysUser, roleIDs []int64) error

	// UpdateWithRoles 更新用户并更新角色（事务）
	UpdateWithRoles(ctx context.Context, userID int64, updates map[string]interface{}, roleIDs []int64) error
}
