package repository

import (
	"context"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
)

// ====================
// 用户管理 Repository
// ====================

// IUserRepository 用户仓储接口
type IUserRepository interface {
	// FindByID 根据 ID 查询用户
	FindByID(ctx context.Context, id int64) (*model.SysUser, error)

	// FindByUsername 根据用户名查询用户
	FindByUsername(ctx context.Context, username string) (*model.SysUser, error)

	// ExistsByUsername 检查用户名是否存在
	ExistsByUsername(ctx context.Context, username string, excludeID ...int64) (bool, error)

	// FindPage 分页查询用户列表
	FindPage(ctx context.Context, q *query.UserPageQuery) (*vo.PageResult[vo.UserPageVO], error)

	// Create 创建用户
	Create(ctx context.Context, user *model.SysUser) error

	// Update 更新用户
	Update(ctx context.Context, user *model.SysUser) error

	// UpdateStatus 更新用户状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// UpdatePassword 更新用户密码
	UpdatePassword(ctx context.Context, id int64, password string) error

	// Delete 删除用户（逻辑删除）
	Delete(ctx context.Context, ids []int64) error

	// FindUserAuthInfo 查询用户认证信息（含角色、权限）
	FindUserAuthInfo(ctx context.Context, username string) (*model.UserAuthInfo, error)

	// AssignRoles 分配用户角色
	AssignRoles(ctx context.Context, userID int64, roleIDs []int64) error

	// GetUserRoleIDs 获取用户角色 ID 列表
	GetUserRoleIDs(ctx context.Context, userID int64) ([]int64, error)

	// GetFormData 获取用户表单数据
	GetFormData(ctx context.Context, userID int64) (*bo.UserFormBO, error)
}

// ====================
// 角色管理 Repository
// ====================

// IRoleRepository 角色仓储接口
type IRoleRepository interface {
	// FindByID 根据 ID 查询角色
	FindByID(ctx context.Context, id int64) (*model.SysRole, error)

	// FindByCode 根据编码查询角色
	FindByCode(ctx context.Context, code string) (*model.SysRole, error)

	// ExistsByCode 检查角色编码是否存在
	ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error)

	// ExistsByName 检查角色名称是否存在
	ExistsByName(ctx context.Context, name string, excludeID ...int64) (bool, error)

	// FindPage 分页查询角色列表
	FindPage(ctx context.Context, q *query.RolePageQuery) (*vo.PageResult[vo.RolePageVO], error)

	// FindOptions 获取角色下拉选项
	FindOptions(ctx context.Context) ([]vo.Option, error)

	// Create 创建角色
	Create(ctx context.Context, role *model.SysRole) error

	// Update 更新角色
	Update(ctx context.Context, role *model.SysRole) error

	// UpdateStatus 更新角色状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// Delete 删除角色（逻辑删除）
	Delete(ctx context.Context, ids []int64) error

	// HasUsers 检查角色是否关联用户
	HasUsers(ctx context.Context, roleID int64) (bool, error)

	// GetMenuIDs 获取角色菜单 ID 列表
	GetMenuIDs(ctx context.Context, roleID int64) ([]int64, error)

	// AssignMenus 分配角色菜单
	AssignMenus(ctx context.Context, roleID int64, menuIDs []int64) error

	// GetFormData 获取角色表单数据
	GetFormData(ctx context.Context, roleID int64) (*bo.RoleFormBO, error)
}

// ====================
// 菜单管理 Repository
// ====================

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

	// FindRoutesByRoles 根据角色获取路由菜单
	FindRoutesByRoles(ctx context.Context, roles []string) ([]model.SysMenu, error)

	// FindPermsByRoles 根据角色获取权限标识列表
	FindPermsByRoles(ctx context.Context, roles []string) ([]string, error)

	// GetOptions 获取菜单下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetFormData 获取菜单表单数据
	GetFormData(ctx context.Context, menuID int64) (*bo.MenuForm, error)
}

// MenuForm 别名，兼容现有命名
type MenuForm = bo.MenuForm

// ====================
// 部门管理 Repository
// ====================

// IDeptRepository 部门仓储接口
type IDeptRepository interface {
	// FindByID 根据 ID 查询部门
	FindByID(ctx context.Context, id int64) (*model.SysDept, error)

	// FindAll 查询所有部门
	FindAll(ctx context.Context, q *query.DeptQuery) ([]model.SysDept, error)

	// FindByParentID 根据父 ID 查询子部门
	FindByParentID(ctx context.Context, parentID int64) ([]model.SysDept, error)

	// Create 创建部门
	Create(ctx context.Context, dept *model.SysDept) error

	// Update 更新部门
	Update(ctx context.Context, dept *model.SysDept) error

	// Delete 删除部门
	Delete(ctx context.Context, id int64) error

	// HasChildren 检查部门是否有子部门
	HasChildren(ctx context.Context, id int64) (bool, error)

	// HasUsers 检查部门是否关联用户
	HasUsers(ctx context.Context, deptID int64) (bool, error)

	// GetOptions 获取部门下拉选项
	GetOptions(ctx context.Context) ([]vo.Option, error)

	// GetFormData 获取部门表单数据
	GetFormData(ctx context.Context, deptID int64) (*bo.DeptFormBO, error)

	// GetSubDeptIDs 获取部门及所有子部门 ID
	GetSubDeptIDs(ctx context.Context, deptID int64) ([]int64, error)
}

// ====================
// 字典管理 Repository
// ====================

// IDictTypeRepository 字典类型仓储接口
type IDictTypeRepository interface {
	// FindByID 根据 ID 查询字典类型
	FindByID(ctx context.Context, id int64) (*model.SysDictType, error)

	// FindByCode 根据编码查询字典类型
	FindByCode(ctx context.Context, code string) (*model.SysDictType, error)

	// ExistsByCode 检查字典类型编码是否存在
	ExistsByCode(ctx context.Context, code string, excludeID ...int64) (bool, error)

	// FindPage 分页查询字典类型
	FindPage(ctx context.Context, q *query.DictTypePageQuery) (*vo.PageResult[vo.DictTypePageVO], error)

	// Create 创建字典类型
	Create(ctx context.Context, dictType *model.SysDictType) error

	// Update 更新字典类型
	Update(ctx context.Context, dictType *model.SysDictType) error

	// Delete 删除字典类型
	Delete(ctx context.Context, ids []int64) error
}

// IDictRepository 字典数据仓储接口
type IDictRepository interface {
	// FindByID 根据 ID 查询字典
	FindByID(ctx context.Context, id int64) (*model.SysDict, error)

	// FindByTypeCode 根据类型编码查询字典列表
	FindByTypeCode(ctx context.Context, typeCode string) ([]model.SysDict, error)

	// FindPage 分页查询字典
	FindPage(ctx context.Context, q *query.DictPageQuery) (*vo.PageResult[vo.DictPageVO], error)

	// Create 创建字典
	Create(ctx context.Context, dict *model.SysDict) error

	// Update 更新字典
	Update(ctx context.Context, dict *model.SysDict) error

	// Delete 删除字典
	Delete(ctx context.Context, ids []int64) error
}

// ====================
// 文件管理 Repository
// ====================

// IFileRepository 文件仓储接口
type IFileRepository interface {
	// FindByID 根据 ID 查询文件
	FindByID(ctx context.Context, id int64) (*model.SysFile, error)

	// FindByMD5 根据 MD5 查询文件
	FindByMD5(ctx context.Context, md5 string) (*model.SysFile, error)

	// FindByObjectName 根据对象名称查询文件
	FindByObjectName(ctx context.Context, objectName string) (*model.SysFile, error)

	// Create 创建文件记录
	Create(ctx context.Context, file *model.SysFile) (*model.SysFile, error)

	// Delete 删除文件记录
	Delete(ctx context.Context, ids []int64) error

	// FindByPath 根据路径查询文件
	FindByPath(ctx context.Context, path string) (*model.SysFile, error)
}

// ====================
// 数据集管理 Repository
// ====================

// IDatasetRepository 数据集仓储接口
type IDatasetRepository interface {
	// FindByID 根据 ID 查询数据集
	FindByID(ctx context.Context, id int64) (*model.SysDataset, error)

	// FindPage 分页查询数据集
	FindPage(ctx context.Context, q *query.DatasetQuery) (*vo.PageResult[vo.DatasetVO], error)

	// Create 创建数据集
	Create(ctx context.Context, dataset *model.SysDataset) error

	// Update 更新数据集
	Update(ctx context.Context, dataset *model.SysDataset) error

	// Delete 删除数据集
	Delete(ctx context.Context, ids []int64) error

	// GetFormData 获取数据集表单数据
	GetFormData(ctx context.Context, datasetID int64) (*bo.DatasetFormBO, error)
}

// IDatasetItemRepository 数据项仓储接口
type IDatasetItemRepository interface {
	// FindByID 根据 ID 查询数据项
	FindByID(ctx context.Context, id int64) (*model.SysDatasetItem, error)

	// FindByDatasetID 根据数据集 ID 查询数据项
	FindByDatasetID(ctx context.Context, datasetID int64) ([]model.SysDatasetItem, error)

	// Create 创建数据项
	Create(ctx context.Context, item *model.SysDatasetItem) error

	// BatchCreate 批量创建数据项
	BatchCreate(ctx context.Context, items []model.SysDatasetItem) error

	// Delete 删除数据项
	Delete(ctx context.Context, ids []int64) error

	// DeleteByDatasetID 根据数据集 ID 删除数据项
	DeleteByDatasetID(ctx context.Context, datasetID int64) error

	// FindPage 分页查询数据项
	FindPage(ctx context.Context, datasetID int64, pageNum, pageSize int) ([]model.SysDatasetItem, int64, error)

	// Update 更新数据项
	Update(ctx context.Context, item *model.SysDatasetItem) error
}

// IItemFileRepository 项文件仓储接口
type IItemFileRepository interface {
	// FindByID 根据 ID 查询项文件
	FindByID(ctx context.Context, id int64) (*model.SysItemFile, error)

	// FindByItemID 根据数据项 ID 查询所有项文件
	FindByItemID(ctx context.Context, itemID int64) ([]model.SysItemFile, error)

	// Create 创建项文件
	Create(ctx context.Context, itemFile *model.SysItemFile) error

	// Update 更新项文件
	Update(ctx context.Context, itemFile *model.SysItemFile) error

	// Delete 删除项文件
	Delete(ctx context.Context, id int64) error

	// DeleteByItemID 根据数据项 ID 删除所有项文件
	DeleteByItemID(ctx context.Context, itemID int64) error

	// UpdateThumbnail 更新缩略图
	UpdateThumbnail(ctx context.Context, itemFileID, thumbnailFileID int64) error
}

// ====================
// 算法管理 Repository
// ====================

// IAlgorithmRepository 算法仓储接口
type IAlgorithmRepository interface {
	// FindByID 根据 ID 查询算法
	FindByID(ctx context.Context, id int64) (*model.SysAlgorithm, error)

	// FindPage 分页查询算法
	FindPage(ctx context.Context, q *query.AlgorithmQuery) (*vo.PageResult[vo.AlgorithmVO], error)

	// FindOptions 获取算法下拉选项
	FindOptions(ctx context.Context) ([]vo.Option, error)

	// Create 创建算法
	Create(ctx context.Context, algorithm *model.SysAlgorithm) error

	// Update 更新算法
	Update(ctx context.Context, algorithm *model.SysAlgorithm) error

	// Delete 删除算法
	Delete(ctx context.Context, ids []int64) error

	// UpdateStatus 更新算法状态
	UpdateStatus(ctx context.Context, id int64, status int8) error
}

// ====================
// 任务管理 Repository
// ====================

// ITaskRepository 任务仓储接口
type ITaskRepository interface {
	// FindByID 根据 ID 查询任务
	FindByID(ctx context.Context, id int64) (*model.SysTask, error)

	// FindByTaskID 根据任务唯一 ID 查询任务
	FindByTaskID(ctx context.Context, taskID string) (*model.SysTask, error)

	// FindPage 分页查询任务
	FindPage(ctx context.Context, q any) (*vo.PageResult[vo.TaskVO], error)

	// Create 创建任务
	Create(ctx context.Context, task *model.SysTask) error

	// Update 更新任务
	Update(ctx context.Context, task *model.SysTask) error

	// UpdateFields 更新任务指定字段
	UpdateFields(ctx context.Context, id int64, fields map[string]interface{}) error

	// UpdateStatus 更新任务状态
	UpdateStatus(ctx context.Context, id int64, status int8) error

	// Delete 删除任务
	Delete(ctx context.Context, ids []int64) error

	// UpdateExpiredTasks 更新过期任务状态
	UpdateExpiredTasks(ctx context.Context, threshold time.Time) (int64, error)

	// CountDatasetItems 统计数据集数据项数量
	CountDatasetItems(ctx context.Context, datasetID int64) (int64, error)

	// CountItemFiles 统计数据项文件数量
	CountItemFiles(ctx context.Context, itemIDs []int64) (int64, error)
}
