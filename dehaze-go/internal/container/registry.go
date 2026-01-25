package container

import (
	"github.com/earthyzinc/dehaze-go/internal/repository"
	"github.com/earthyzinc/dehaze-go/internal/service"
	"github.com/earthyzinc/dehaze-go/pkg/cache"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
)

// RegisterRepositories 注册所有 Repository 到容器
func (c *Container) RegisterRepositories() {
	db := c.DB()

	// 用户管理
	c.Register("repository", "user", repository.NewUserRepository(db))

	// 角色管理
	c.Register("repository", "role", repository.NewRoleRepository(db))

	// 菜单管理
	c.Register("repository", "menu", repository.NewMenuRepository(db))

	// 部门管理
	c.Register("repository", "dept", repository.NewDeptRepository(db))

	// 字典管理
	c.Register("repository", "dictType", repository.NewDictTypeRepository(db))
	c.Register("repository", "dict", repository.NewDictRepository(db))

	// 算法管理
	c.Register("repository", "algorithm", repository.NewAlgorithmRepository(db))

	// 数据集管理
	c.Register("repository", "dataset", repository.NewDatasetRepository(db))

	// 数据项管理
	c.Register("repository", "datasetItem", repository.NewDatasetItemRepository(db))

	// 项文件管理
	c.Register("repository", "itemFile", repository.NewItemFileRepository(db))

	// 文件管理
	c.Register("repository", "file", repository.NewFileRepository(db))

	// 任务管理
	c.Register("repository", "task", repository.NewTaskRepository(db))
}

// RegisterCaches 注册所有 Cache 到容器
func (c *Container) RegisterCaches() {
	c.Register("cache", "common", cache.GetCache())
}

// RegisterServices 注册所有 Service 到容器
// 注意：Service 依赖 Repository 和 Cache，需要在它们注册之后调用
func (c *Container) RegisterServices() {
	// 获取依赖的 Repository
	userRepo := MustGetRepository[repository.IUserRepository](c, "user")
	roleRepo := MustGetRepository[repository.IRoleRepository](c, "role")
	menuRepo := MustGetRepository[repository.IMenuRepository](c, "menu")
	deptRepo := MustGetRepository[repository.IDeptRepository](c, "dept")
	dictTypeRepo := MustGetRepository[repository.IDictTypeRepository](c, "dictType")
	dictRepo := MustGetRepository[repository.IDictRepository](c, "dict")
	algorithmRepo := MustGetRepository[repository.IAlgorithmRepository](c, "algorithm")
	datasetRepo := MustGetRepository[repository.IDatasetRepository](c, "dataset")
	datasetItemRepo := MustGetRepository[repository.IDatasetItemRepository](c, "datasetItem")
	itemFileRepo := MustGetRepository[repository.IItemFileRepository](c, "itemFile")
	fileRepo := MustGetRepository[repository.IFileRepository](c, "file")
	taskRepo := MustGetRepository[repository.ITaskRepository](c, "task")

	// 获取 Logger
	logger := c.Logger()

	// 注册 Service
	c.Register("service", "user", service.NewUserService(userRepo, roleRepo))

	// TODO: 注册其他 Service（需要先改造 Service 层）
	// c.Register("service", "auth", service.NewAuthService(userRepo, userCache, tokenCache, captchaCache))
	c.Register("service", "role", service.NewRoleService(roleRepo, menuRepo))
	c.Register("service", "menu", service.NewMenuService(menuRepo))
	c.Register("service", "dept", service.NewDeptService(deptRepo))
	c.Register("service", "dictType", service.NewDictTypeService(dictTypeRepo))
	c.Register("service", "dict", service.NewDictService(dictRepo))
	c.Register("service", "algorithm", service.NewAlgorithmService(algorithmRepo))
	c.Register("service", "dataset", service.NewDatasetService(datasetRepo))
	c.Register("service", "datasetItem", service.NewDatasetItemService(datasetItemRepo))
	c.Register("service", "itemFile", service.NewItemFileService(itemFileRepo))
	c.Register("service", "datasetOperation", service.NewDatasetOperationService(datasetRepo, datasetItemRepo, itemFileRepo))
	c.Register("service", "file", service.NewSysFileService(fileRepo))

	// 任务服务（需要配置）
	if cfg := c.Config(); cfg != nil {
		c.Register("service", "task", service.NewTaskService(taskRepo, datasetRepo, *c.CommonCache(), logger, cfg))
	}

}

// InitAll 初始化所有组件
func (c *Container) InitAll() {
	c.RegisterRepositories()
	c.RegisterCaches()
	c.RegisterServices()
}

// ============================================
// 便捷获取方法
// ============================================

// UserRepository 获取用户仓储
func (c *Container) UserRepository() repository.IUserRepository {
	return MustGetRepository[repository.IUserRepository](c, "user")
}

// RoleRepository 获取角色仓储
func (c *Container) RoleRepository() repository.IRoleRepository {
	return MustGetRepository[repository.IRoleRepository](c, "role")
}

// MenuRepository 获取菜单仓储
func (c *Container) MenuRepository() repository.IMenuRepository {
	return MustGetRepository[repository.IMenuRepository](c, "menu")
}

// DeptRepository 获取部门仓储
func (c *Container) DeptRepository() repository.IDeptRepository {
	return MustGetRepository[repository.IDeptRepository](c, "dept")
}

// DictTypeRepository 获取字典类型仓储
func (c *Container) DictTypeRepository() repository.IDictTypeRepository {
	return MustGetRepository[repository.IDictTypeRepository](c, "dictType")
}

// DictRepository 获取字典数据仓储
func (c *Container) DictRepository() repository.IDictRepository {
	return MustGetRepository[repository.IDictRepository](c, "dict")
}

// AlgorithmRepository 获取算法仓储
func (c *Container) AlgorithmRepository() repository.IAlgorithmRepository {
	return MustGetRepository[repository.IAlgorithmRepository](c, "algorithm")
}

// DatasetRepository 获取数据集仓储
func (c *Container) DatasetRepository() repository.IDatasetRepository {
	return MustGetRepository[repository.IDatasetRepository](c, "dataset")
}

// DatasetItemRepository 获取数据项仓储
func (c *Container) DatasetItemRepository() repository.IDatasetItemRepository {
	return MustGetRepository[repository.IDatasetItemRepository](c, "datasetItem")
}

// ItemFileRepository 获取项文件仓储
func (c *Container) ItemFileRepository() repository.IItemFileRepository {
	return MustGetRepository[repository.IItemFileRepository](c, "itemFile")
}

// FileRepository 获取文件仓储
func (c *Container) FileRepository() repository.IFileRepository {
	return MustGetRepository[repository.IFileRepository](c, "file")
}

// UserService 获取用户服务
func (c *Container) UserService() service.IUserService {
	return MustGetService[service.IUserService](c, "user")
}

// RoleService 获取角色服务
func (c *Container) RoleService() service.IRoleService {
	return MustGetService[service.IRoleService](c, "role")
}

// MenuService 获取菜单服务
func (c *Container) MenuService() service.IMenuService {
	return MustGetService[service.IMenuService](c, "menu")
}

// DeptService 获取部门服务
func (c *Container) DeptService() service.IDeptService {
	return MustGetService[service.IDeptService](c, "dept")
}

// DictTypeService 获取字典类型服务
func (c *Container) DictTypeService() service.IDictTypeService {
	return MustGetService[service.IDictTypeService](c, "dictType")
}

// DictService 获取字典数据服务
func (c *Container) DictService() service.IDictService {
	return MustGetService[service.IDictService](c, "dict")
}

// AlgorithmService 获取算法服务
func (c *Container) AlgorithmService() service.IAlgorithmService {
	return MustGetService[service.IAlgorithmService](c, "algorithm")
}

// DatasetService 获取数据集服务
func (c *Container) DatasetService() service.IDatasetService {
	return MustGetService[service.IDatasetService](c, "dataset")
}

// DatasetItemService 获取数据项服务
func (c *Container) DatasetItemService() *service.DatasetItemService {
	return MustGetService[*service.DatasetItemService](c, "datasetItem")
}

// ItemFileService 获取项文件服务
func (c *Container) ItemFileService() *service.ItemFileService {
	return MustGetService[*service.ItemFileService](c, "itemFile")
}

// DatasetOperationService 获取数据集操作服务
func (c *Container) DatasetOperationService() *service.DatasetOperationService {
	return MustGetService[*service.DatasetOperationService](c, "datasetOperation")
}

// SysFileService 获取文件服务
func (c *Container) SysFileService() *service.SysFileService {
	return MustGetService[*service.SysFileService](c, "file")
}

// TaskRepository 获取任务仓储
func (c *Container) TaskRepository() repository.ITaskRepository {
	return MustGetRepository[repository.ITaskRepository](c, "task")
}

// CommonCache 获取通用缓存
func (c *Container) CommonCache() *types.ICache {
	return MustGetCache[*types.ICache](c, "common")
}

// TaskService 获取任务服务
func (c *Container) TaskService() *service.TaskService {
	return MustGetService[*service.TaskService](c, "task")
}
