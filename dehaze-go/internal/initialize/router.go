package initialize

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/internal/container"
	"github.com/earthyzinc/dehaze-go/internal/global"
	"github.com/earthyzinc/dehaze-go/internal/router"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/logger"
	"github.com/gin-gonic/gin"
)

func Routers() {
	Router := gin.New()
	Router.Use(gin.Recovery())
	Router.Use(gin.Logger())

	Router.GET("/health", func(c *gin.Context) {
		common.Ok(c)
	})

	// 初始化容器并注册服务依赖
	ctr := container.GetInstance()
	ctr.InitAll()

	// 初始化 API 服务依赖
	api.InitServices(ctr)

	PublicGroup := Router.Group("/api/v1")
	{
		router.RouterGroupApp.InitAuthRouter(PublicGroup)
		router.RouterGroupApp.InitSysUserRouter(PublicGroup)
		router.RouterGroupApp.InitSysRoleRouter(PublicGroup)
		router.RouterGroupApp.InitSysDeptRouter(PublicGroup)
		router.RouterGroupApp.InitSysDictRouter(PublicGroup)
		router.RouterGroupApp.InitDatasetRouter(PublicGroup)
		router.RouterGroupApp.InitDatasetOperationRouter(PublicGroup)
		router.RouterGroupApp.InitFileRouter(PublicGroup)
		router.RouterGroupApp.InitDatasetItemRouter(PublicGroup)
		router.RouterGroupApp.InitItemFileRouter(PublicGroup)
		router.RouterGroupApp.InitAlgorithmRouter(PublicGroup)
	}

	global.GIN = Router
	global.ROUTES = Router.Routes()

	logger.Info("路由初始化成功")

}
