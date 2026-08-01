package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterRecommendationRoutes(rg *gin.RouterGroup, recApi *api.RecommendationApi) {
	recRouter := rg.Group("/recommendations")
	{
		// 登录用户接口
		recRouter.POST("/analyze", recApi.Analyze)
		recRouter.GET("/algorithms", recApi.GetAlgorithmRecommendations)
		recRouter.POST("/feedback", recApi.SubmitFeedback)

		// 管理员接口
		recRouter.GET("/rules", middleware.Permission("sys:recommendation:rule:view"), recApi.GetRules)
		recRouter.PUT("/rules", middleware.Permission("sys:recommendation:rule:edit"), recApi.UpdateRule)
		recRouter.GET("/report", middleware.Permission("sys:recommendation:report"), recApi.GetReport)
	}
}
