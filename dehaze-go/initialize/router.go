package initialize

import (
	"net/http"

	"github.com/earthyzinc/dehaze-go/global"
	"github.com/earthyzinc/dehaze-go/router"
	"github.com/gin-gonic/gin"
)

func Routers() {
	Router := gin.New()
	Router.Use(gin.Recovery())
	Router.Use(gin.Logger())

	Router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"code": 0, "message": "ok"})
	})

	PublicGroup := Router.Group("/api/v1")
	{
		router.RouterGroupApp.InitAuthRouter(PublicGroup)
		router.RouterGroupApp.InitSysUserRouter(PublicGroup)
		router.RouterGroupApp.InitSysRoleRouter(PublicGroup)
		router.RouterGroupApp.InitSysDeptRouter(PublicGroup)
		router.RouterGroupApp.InitSysDictRouter(PublicGroup)
		router.RouterGroupApp.InitDatasetRouter(PublicGroup)
		router.RouterGroupApp.InitFileRouter(PublicGroup)
		router.RouterGroupApp.InitDatasetItemRouter(PublicGroup)
		router.RouterGroupApp.InitItemFileRouter(PublicGroup)
	}

	global.GIN = Router
	global.ROUTES = Router.Routes()

	global.LOG.Info("路由初始化成功")

}
