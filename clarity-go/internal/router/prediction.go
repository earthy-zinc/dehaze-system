package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
	"github.com/gin-gonic/gin"
)

// RegisterPredictionRoutes 图像任务路由（设计 §9.2：/api/v1/images/*）
func RegisterPredictionRoutes(rg *gin.RouterGroup, predictionApi *api.SysPredictionApi) {
	imagesGroup := rg.Group("/images")

	imagesGroup.POST("/edits", predictionApi.SubmitImageEdit)                // 提交图像编辑任务
	imagesGroup.GET("/tasks", predictionApi.ListImageTasks)                  // 图像任务历史列表
	imagesGroup.GET("/tasks/:taskId", predictionApi.GetImageTask)            // 查询图像任务状态
	imagesGroup.POST("/tasks/:taskId/cancel", predictionApi.CancelImageTask) // 取消图像任务（幂等，仅本人可取消）
}
