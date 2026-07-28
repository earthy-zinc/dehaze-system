package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterFeedbackRoutes 注册反馈评价模块路由
// 注意：literal 路径必须注册在 /{id} 之前以避免路由冲突
func RegisterFeedbackRoutes(rg *gin.RouterGroup, feedbackApi *api.FeedbackApi) {
	// ============ 评价接口（/feedback/ratings） ============
	ratingRouter := rg.Group("/ratings")
	{
		ratingRouter.POST("", feedbackApi.CreateRating)
		// literal 路径优先注册
		ratingRouter.GET("/my", feedbackApi.ListMyRatings)
		ratingRouter.GET("/by-prediction/:predictionLogId", feedbackApi.GetRatingByPrediction)
		ratingRouter.GET("/page", middleware.Permission("feedback:rating:list"), feedbackApi.ListRatings)
		ratingRouter.GET("/stats", middleware.Permission("feedback:stats"), feedbackApi.GetRatingStats)
		// 带参数路径最后注册
		ratingRouter.PUT("/:id", feedbackApi.UpdateRating)
		ratingRouter.PUT("/:id/hide", middleware.Permission("feedback:rating:edit"), feedbackApi.HideRating)
		ratingRouter.POST("/:id/reply", middleware.Permission("feedback:rating:reply"), feedbackApi.ReplyRating)
	}

	// ============ 反馈接口（/feedback） ============
	// literal 路径优先注册
	rg.POST("", feedbackApi.CreateFeedback)
	rg.GET("/my", feedbackApi.ListMyFeedback)
	rg.GET("/page", middleware.Permission("feedback:list"), feedbackApi.ListFeedback)
	rg.GET("/stats", middleware.Permission("feedback:stats"), feedbackApi.GetFeedbackStats)
	// 带参数路径最后注册
	rg.GET("/:id", feedbackApi.GetFeedbackDetail)
	rg.POST("/:id/supplement", feedbackApi.SupplementFeedback)
	rg.PUT("/:id/assign", middleware.Permission("feedback:assign"), feedbackApi.AssignFeedback)
	rg.POST("/:id/reply", middleware.Permission("feedback:reply"), feedbackApi.ReplyFeedback)
	rg.PUT("/:id/close", middleware.Permission("feedback:close"), feedbackApi.CloseFeedback)
	rg.PUT("/:id/tags", middleware.Permission("feedback:edit"), feedbackApi.UpdateFeedbackTags)
}
