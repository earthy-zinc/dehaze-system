package router

import "github.com/earthyzinc/dehaze-go/api"

type RouterGroup struct {
	AuthRouter
	SysUserRouter
	SysRoleRouter
	SysDeptRouter
	SysDictRouter
	AlgorithmRouter
	DatasetRouter
	FileRouter
	DatasetItemRouter
	ItemFileRouter
}

var RouterGroupApp = new(RouterGroup)

var (
	authApi        = api.ApiGroupApp.AuthApi
	sysDictApi     = api.ApiGroupApp.SysDictApi
	algorithmApi   = api.ApiGroupApp.AlgorithmApi
	datasetApi     = api.ApiGroupApp.SysDatasetApi
	fileApi        = api.ApiGroupApp.SysFileApi
	datasetItemApi = api.ApiGroupApp.SysDatasetItemApi
	itemFileApi    = api.ApiGroupApp.SysItemFileApi
)
