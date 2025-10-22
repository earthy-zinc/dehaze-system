package service

type ServiceGroup struct {
	UserService
	UserServiceExtend
	RoleService
	DeptService
	DictService
	DictTypeService
	MenuService
	AlgorithmService
	DatasetService
	SysFileService
	DatasetItemService
	ItemFileService
}

var ServiceGroupApp = new(ServiceGroup)