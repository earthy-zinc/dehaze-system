package service

import (
	"context"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/dto"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ====================
// 认证服务接口
// ====================

// IAuthService 认证服务接口
type IAuthService interface {
	// Login 用户登录
	Login(ctx context.Context, req *bo.LoginRequest) (*dto.LoginResult, error)

	// Logout 用户注销
	Logout(ctx context.Context, token string) error

	// RefreshToken 刷新令牌
	RefreshToken(ctx context.Context, refreshToken string) (*dto.LoginResult, error)

	// GetCaptcha 获取验证码
	GetCaptcha(ctx context.Context) (*dto.CaptchaResult, error)

	// GetCurrentUserInfo 获取当前用户信息
	GetCurrentUserInfo(ctx context.Context, userID int64) (*dto.UserAuthInfo, error)
}

// ====================
// 用户管理服务接口
// ====================

// IUserService 用户服务接口
type IUserService interface {
	// Login 用户登录
	Login(ctx context.Context, u *model.SysUser) (*model.UserAuthInfo, error)

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

	// ImportUsers 导入用户
	ImportUsers(ctx context.Context, data []vo.UserImportVO) (*vo.ImportResultVO, error)

	// ExportUsers 导出用户
	ExportUsers(ctx context.Context, q *query.UserPageQuery) ([]vo.UserExportVO, error)
}

// ====================
// 角色管理服务接口
// ====================

// IRoleService 角色服务接口
type IRoleService interface {
	// GetPage 角色分页列表
	GetPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error)

	// GetOptions 获取角色下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetFormData 获取角色表单数据
	GetFormData(ctx context.Context, id int64) (*bo.RoleFormBO, error)

	// Create 创建角色
	Create(ctx context.Context, form *bo.RoleFormBO) error

	// Update 更新角色
	Update(ctx context.Context, id int64, form *bo.RoleFormBO) error

	// Delete 删除角色（支持批量）
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新角色状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// GetMenuIDs 获取角色菜单 ID 集合
	GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error)

	// AssignMenus 分配菜单权限
	AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error
}

// ====================
// 菜单管理服务接口
// ====================

// IMenuService 菜单服务接口
type IMenuService interface {
	// GetList 获取菜单列表
	GetList(ctx context.Context, q *query.MenuQuery) ([]vo.MenuVO, error)

	// GetFormData 获取菜单表单数据
	GetFormData(ctx context.Context, id int64) (*bo.MenuForm, error)

	// Create 创建菜单
	Create(ctx context.Context, form *bo.MenuForm) error

	// Update 更新菜单
	Update(ctx context.Context, id int64, form *bo.MenuForm) error

	// Delete 删除菜单
	Delete(ctx context.Context, id int64) error

	// GetOptions 获取菜单下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetRoutes 获取当前用户路由菜单
	GetRoutes(ctx context.Context, roles []string) ([]vo.RouteVO, error)
}

// ====================
// 部门管理服务接口
// ====================

// IDeptService 部门服务接口
type IDeptService interface {
	// GetList 获取部门列表
	GetList(ctx context.Context, q *query.DeptQuery) ([]vo.DeptVO, error)

	// GetFormData 获取部门表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DeptFormBO, error)

	// Create 创建部门
	Create(ctx context.Context, form *bo.DeptFormBO) error

	// Update 更新部门
	Update(ctx context.Context, id int64, form *bo.DeptFormBO) error

	// Delete 删除部门
	Delete(ctx context.Context, id int64) error

	// GetOptions 获取部门下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)
}

// ====================
// 字典管理服务接口
// ====================

// IDictTypeService 字典类型服务接口
type IDictTypeService interface {
	// GetPage 字典类型分页列表
	GetPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error)

	// GetFormData 获取字典类型表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DictTypeFormBO, error)

	// Create 创建字典类型
	Create(ctx context.Context, form *bo.DictTypeFormBO) error

	// Update 更新字典类型
	Update(ctx context.Context, id int64, form *bo.DictTypeFormBO) error

	// Delete 删除字典类型
	Delete(ctx context.Context, ids []int64) error
}

// IDictService 字典数据服务接口
type IDictService interface {
	// GetPage 字典数据分页列表
	GetPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error)

	// GetByTypeCode 根据类型编码获取字典列表
	GetByTypeCode(ctx context.Context, typeCode string) ([]vo.Option, error)

	// GetFormData 获取字典表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DictFormBO, error)

	// Create 创建字典
	Create(ctx context.Context, form *bo.DictFormBO) error

	// Update 更新字典
	Update(ctx context.Context, id int64, form *bo.DictFormBO) error

	// Delete 删除字典
	Delete(ctx context.Context, ids []int64) error
}

// ====================
// 文件管理服务接口
// ====================

// IFileService 文件服务接口
type IFileService interface {
	// Upload 上传文件
	Upload(ctx context.Context, file *bo.FileBO) (*dto.FileInfo, error)

	// Delete 删除文件
	Delete(ctx context.Context, ids []int64) error

	// GetByID 根据 ID 获取文件信息
	GetByID(ctx context.Context, id int64) (*dto.FileInfo, error)
}

// ====================
// 数据集管理服务接口
// ====================

// IDatasetService 数据集服务接口
type IDatasetService interface {
	// GetPage 数据集分页列表
	GetPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error)

	// GetFormData 获取数据集表单数据
	GetFormData(ctx context.Context, id int64) (*bo.DatasetFormBO, error)

	// Create 创建数据集
	Create(ctx context.Context, form *bo.DatasetFormBO) error

	// Update 更新数据集
	Update(ctx context.Context, id int64, form *bo.DatasetFormBO) error

	// Delete 删除数据集
	Delete(ctx context.Context, ids []int64) error
}

// ====================
// 算法管理服务接口
// ====================

// IAlgorithmService 算法服务接口
type IAlgorithmService interface {
	// GetPage 算法分页列表
	GetPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error)

	// GetOptions 获取算法下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetFormData 获取算法表单数据
	GetFormData(ctx context.Context, id int64) (*bo.AlgorithmFormBO, error)

	// Create 创建算法
	Create(ctx context.Context, form *bo.AlgorithmFormBO) error

	// Update 更新算法
	Update(ctx context.Context, id int64, form *bo.AlgorithmFormBO) error

	// Delete 删除算法
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新算法状态
	UpdateStatus(ctx context.Context, id int64, status int8) error
}

// ====================
// 任务管理服务接口
// ====================

// ITaskService 任务服务接口
type ITaskService interface {
	// GetPage 任务分页列表
	GetPage(ctx context.Context, q any) (*vo.PageResult[vo.TaskVO], error)

	// GetByID 根据 ID 获取任务详情
	GetByID(ctx context.Context, id int64) (*vo.TaskDetailVO, error)

	// Create 创建任务
	Create(ctx context.Context, form *bo.TaskBO) (int64, error)

	// Delete 删除任务
	Delete(ctx context.Context, ids []int64) error

	// Cancel 取消任务
	Cancel(ctx context.Context, id int64) error
}
