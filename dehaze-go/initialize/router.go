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
	}

	global.GIN = Router
	global.ROUTES = Router.Routes()

	global.LOG.Info("路由初始化成功")

}
