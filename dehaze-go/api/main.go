package api

import "github.com/earthyzinc/dehaze-go/service"

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
}

var ApiGroupApp = new(ApiGroup)

var (
	userService         = service.ServiceGroupApp.UserService
	sysUserApi          = service.ServiceGroupApp.UserServiceExtend
	roleService         = service.ServiceGroupApp.RoleService
	deptService         = service.ServiceGroupApp.DeptService
	dictService         = service.ServiceGroupApp.DictService
	dictTypeService     = service.ServiceGroupApp.DictTypeService
	menuService         = service.ServiceGroupApp.MenuService
	algorithmService    = service.ServiceGroupApp.AlgorithmService
	datasetService      = service.ServiceGroupApp.DatasetService
	sysFileService      = service.ServiceGroupApp.SysFileService
	datasetItemService  = service.ServiceGroupApp.DatasetItemService
	itemFileService     = service.ServiceGroupApp.ItemFileService
)