package api

import (
	"github.com/earthyzinc/dehaze-go/internal/container"
	"github.com/earthyzinc/dehaze-go/internal/service"
)

type ApiGroup struct {
	AuthApi
	SysUserApi
	SysRoleApi
	SysDeptApi
	SysDictApi
	SysMenuApi
	AlgorithmApi
	SysDatasetApi
	SysFileApi
	SysDatasetItemApi
	SysItemFileApi
	DatasetOperationApi
}

var ApiGroupApp = new(ApiGroup)

// 全局容器引用（由 InitServices 设置）
var globalContainer *container.Container

// 服务获取函数（通过容器获取）
func getUserService() service.IUserService {
	return globalContainer.UserService()
}

// InitServices 初始化所有 API 的服务依赖
func InitServices(c *container.Container) {
	globalContainer = c

	// 初始化各 API 模块的服务依赖
	ApiGroupApp.AlgorithmApi.algorithmService = c.AlgorithmService()
	ApiGroupApp.SysRoleApi.roleService = c.RoleService()
	ApiGroupApp.SysDeptApi.deptService = c.DeptService()
	ApiGroupApp.SysDictApi.dictService = c.DictService()
	ApiGroupApp.SysDictApi.dictTypeService = c.DictTypeService()
	ApiGroupApp.SysMenuApi.menuService = c.MenuService().(*service.MenuService)
	ApiGroupApp.SysDatasetApi.datasetService = c.DatasetService().(*service.DatasetService)
	ApiGroupApp.SysDatasetItemApi.datasetItemService = *c.DatasetItemService()
	ApiGroupApp.SysItemFileApi.itemFileService = *c.ItemFileService()
	ApiGroupApp.SysFileApi.sysFileService = *c.SysFileService()
	ApiGroupApp.DatasetOperationApi.operationService = c.DatasetOperationService()
}
