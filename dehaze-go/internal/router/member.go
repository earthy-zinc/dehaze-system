package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterMemberRoutes(rg *gin.RouterGroup, memberApi *api.MemberApi) {
	memberRouter := rg.Group("/members")
	{
		memberRouter.GET("/profile", memberApi.GetProfile)
		memberRouter.GET("/growth-logs", memberApi.GetGrowthLogs)
		memberRouter.POST("/sign-in", memberApi.SignIn)
		memberRouter.GET("/sign-in/calendar", memberApi.GetSignInCalendar)
		memberRouter.GET("/page", memberApi.GetPage)
		memberRouter.GET("/benefits", memberApi.ListBenefits)
		memberRouter.PUT("/benefits/:levelCode", middleware.Permission("member:benefit:edit"), memberApi.UpdateBenefit)
		memberRouter.GET("/:userId", memberApi.GetDetail)
		memberRouter.PUT("/:userId/level", middleware.Permission("member:level:edit"), memberApi.AdjustLevel)
		memberRouter.PUT("/:userId/growth", middleware.Permission("member:growth:edit"), memberApi.AdjustGrowth)
		memberRouter.PUT("/:userId/status", middleware.Permission("member:status:edit"), memberApi.UpdateStatus)
	}
}
