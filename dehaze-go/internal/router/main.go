package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func Router() *gin.Engine {
	r := gin.New()

	// 公开路由
	// PublicGroup := r.Group("")
	// {
	// 	// TODO: 添加 SwaggerHandler 和 HealthCheck
	// 	// PublicGroup.GET("/swagger/*any", api.SwaggerHandler)
	// 	// PublicGroup.GET("/health", api.HealthCheck)
	// }

	// 需要认证的路由
	AuthGroup := r.Group("")
	// AuthGroup.Use(middleware.JWTAuth())

	{
		// 用户相关
		// AuthGroup.POST("/api/v1/user/login", api.LoginApi{}.Login)
		// AuthGroup.GET("/api/v1/user/info", api.UserInfoApi{}.GetUserInfo)
		// AuthGroup.POST("/api/v1/user/logout", api.UserInfoApi{}.Logout)

		// 数据集相关
		AuthGroup.GET("/api/v1/dataset/list", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetApi.GetDatasetList(c)
		})
		// AuthGroup.GET("/api/v1/dataset/options", api.DatasetApi{}.GetDatasetOptions)
		// AuthGroup.POST("/api/v1/dataset", api.DatasetApi{}.CreateDataset)
		AuthGroup.PUT("/api/v1/dataset/:id", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetApi.UpdateDataset(c)
		})
		AuthGroup.DELETE("/api/v1/dataset/:id", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetApi.DeleteDatasets(c)
		})

		// 数据集操作相关（RESTful规范）
		AuthGroup.POST("/api/v1/dataset/operations/items", func(c *gin.Context) {
			api.ApiGroupApp.DatasetOperationApi.CreateDatasetItemWithImages(c)
		})
		AuthGroup.POST("/api/v1/dataset/operations/items/batch", func(c *gin.Context) {
			api.ApiGroupApp.DatasetOperationApi.BatchCreateDatasetItemsWithImages(c)
		})
		AuthGroup.DELETE("/api/v1/dataset/operations/items/:itemId", func(c *gin.Context) {
			api.ApiGroupApp.DatasetOperationApi.DeleteDatasetItemCascade(c)
		})
		AuthGroup.POST("/api/v1/dataset/operations/batch", func(c *gin.Context) {
			api.ApiGroupApp.DatasetOperationApi.BatchDeleteDatasets(c)
		})

		// 数据集项相关
		// AuthGroup.GET("/api/v1/dataset/:id/items", api.DatasetItemApi{}.GetDatasetItemsByDatasetID)
		// AuthGroup.GET("/api/v1/dataset/item/:id", api.DatasetItemApi{}.GetDatasetItemById)
		AuthGroup.POST("/api/v1/dataset/item", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetItemApi.CreateDatasetItem(c)
		})
		AuthGroup.PUT("/api/v1/dataset/item/:id", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetItemApi.UpdateDatasetItem(c)
		})
		AuthGroup.DELETE("/api/v1/dataset/item/:id", func(c *gin.Context) {
			api.ApiGroupApp.SysDatasetItemApi.DeleteDatasetItem(c)
		})

		// 项文件相关
		// AuthGroup.GET("/api/v1/dataset/item/:id/files", api.ItemFileApi{}.GetImageUrlVOs)
		// AuthGroup.GET("/api/v1/dataset/file/:id", api.ItemFileApi{}.GetItemFileById)
		// AuthGroup.DELETE("/api/v1/dataset/file/:id", api.ItemFileApi{}.DeleteItemFile)

		// 文件相关
		AuthGroup.POST("/api/v1/file/upload", func(c *gin.Context) {
			api.ApiGroupApp.SysFileApi.UploadFile(c)
		})
		// AuthGroup.GET("/api/v1/file/:id", api.FileApi{}.GetFileById)
		AuthGroup.DELETE("/api/v1/file/:id", func(c *gin.Context) {
			api.ApiGroupApp.SysFileApi.DeleteFile(c)
		})

		// 算法相关
		// AuthGroup.GET("/api/v1/algorithm/list", api.AlgorithmApi{}.GetAlgorithmList)
		// AuthGroup.POST("/api/v1/algorithm", api.AlgorithmApi{}.CreateAlgorithm)
		// AuthGroup.PUT("/api/v1/algorithm/:id", api.AlgorithmApi{}.UpdateAlgorithm)
		// AuthGroup.DELETE("/api/v1/algorithm/:id", api.AlgorithmApi{}.DeleteAlgorithm)

		// 字典相关
		// AuthGroup.GET("/api/v1/dict/types", api.DictTypeApi{}.GetDictTypeList)
		// AuthGroup.GET("/api/v1/dict/data/:dictType", api.DictApi{}.GetDictDataByDictType)

		// 菜单相关
		// AuthGroup.GET("/api/v1/menu/list", api.MenuApi{}.GetMenuList)
		// AuthGroup.GET("/api/v1/menu/tree", api.MenuApi{}.GetMenuTree)

		// 部门相关
		// AuthGroup.GET("/api/v1/dept/list", api.DeptApi{}.GetDeptList)
		// AuthGroup.GET("/api/v1/dept/tree", api.DeptApi{}.GetDeptTree)

		// 角色相关
		// AuthGroup.GET("/api/v1/role/list", api.RoleApi{}.GetRoleList)
		// AuthGroup.POST("/api/v1/role", api.RoleApi{}.CreateRole)
		// AuthGroup.PUT("/api/v1/role/:id", api.RoleApi{}.UpdateRole)
		// AuthGroup.DELETE("/api/v1/role/:id", api.RoleApi{}.DeleteRole)
	}

	return r
}

type RouterGroup struct {
	AuthRouter
	SysUserRouter
	SysRoleRouter
	SysDeptRouter
	SysDictRouter
	AlgorithmRouter
	DatasetRouter
	DatasetOperationRouter
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
