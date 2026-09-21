package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterPromotionRoutes(rg *gin.RouterGroup, promotionApi *api.PromotionApi) {
	promotionRouter := rg.Group("/packages/promotions")
	{
		promotionRouter.GET("/page", promotionApi.GetPage)
		promotionRouter.POST("", middleware.Permission("package:promotion:add"), promotionApi.Add)
		promotionRouter.PUT("/:id", middleware.Permission("package:promotion:edit"), promotionApi.Update)
		promotionRouter.PUT("/:id/status", middleware.Permission("package:promotion:edit"), promotionApi.UpdateStatus)
		promotionRouter.PUT("/:id/packages", middleware.Permission("package:promotion:edit"), promotionApi.BindPackages)
		promotionRouter.DELETE("/:id", middleware.Permission("package:promotion:delete"), promotionApi.Delete)
	}
}
