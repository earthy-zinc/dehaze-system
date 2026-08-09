package user

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// IUserService 用户服务接口
type IUserService interface {
	// Login 用户登录
	Login(ctx context.Context, u *model.SysUser) (*model.UserAuthInfo, error)

	// GetUserAuthInfo 根据用户名获取认证信息
	GetUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error)

	// GetUserAuthInfoByID 根据用户ID获取认证信息
	GetUserAuthInfoByID(ctx context.Context, userID int64) (*model.UserAuthInfo, error)

	// GetPage 用户分页列表
	GetPage(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error)

	// GetByID 根据 ID 获取用户
	GetByID(ctx context.Context, id int64) (*vo.UserPageVO, error)

	// GetFormData 获取用户表单数据
	GetFormData(ctx context.Context, id int64) (*bo.UserFormBO, error)

	// Create 创建用户
	Create(ctx context.Context, form *bo.UserFormBO) error

	// Update 更新用户
	Update(ctx context.Context, id int64, form *bo.UserFormBO) error

	// Delete 删除用户（支持批量）
	Delete(ctx context.Context, ids []int64) error

	// ResetPassword 重置用户密码
	ResetPassword(ctx context.Context, id int64) error

	// UpdateStatus 更新用户状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// GetCurrentUserInfo 获取当前登录用户信息
	GetCurrentUserInfo(ctx context.Context, userID int64) (*vo.UserInfoVO, error)

	// UpdatePassword 修改用户密码
	UpdatePassword(ctx context.Context, id int64, password string) error

	// Register 用户注册（创建用户并分配 GUEST 角色），返回创建的用户及 GUEST 角色的 dataScope
	Register(ctx context.Context, username, nickname, password string) (*model.SysUser, int8, error)
}
