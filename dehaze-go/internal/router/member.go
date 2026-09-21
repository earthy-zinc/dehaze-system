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
		// 静态子路径必须注册在 /:userId 之前：否则 "benefit-summary" 会被 /:userId 捕获，
		// 表现为「用户ID格式不正确」（路由顺序守卫见 member_route_order_test.go）
		memberRouter.GET("/benefit-summary", memberApi.GetBenefitSummary)
		memberRouter.GET("/trial-status", memberApi.GetTrialStatus)
		memberRouter.GET("/growth-logs", memberApi.GetGrowthLogs)
		memberRouter.POST("/sign-in", memberApi.SignIn)
		memberRouter.GET("/sign-in/calendar", memberApi.GetSignInCalendar)
		memberRouter.GET("/page", middleware.Permission("member:list"), memberApi.GetPage)
		memberRouter.GET("/benefits", memberApi.ListBenefits)
		memberRouter.PUT("/benefits/:levelCode", middleware.Permission("member:benefit:edit"), memberApi.UpdateBenefit)
		memberRouter.GET("/:userId/benefit-usage", middleware.Permission("member:list"), memberApi.GetMemberBenefitUsage)
		memberRouter.GET("/:userId/operation-logs", middleware.Permission("member:list"), memberApi.GetMemberOperationLogs)
		memberRouter.GET("/:userId/growth-logs", middleware.Permission("member:list"), memberApi.GetMemberGrowthLogs)
		memberRouter.GET("/:userId/consumption-records", middleware.Permission("member:list"), memberApi.GetMemberConsumptionRecords)
		memberRouter.GET("/:userId", memberApi.GetDetail)
		memberRouter.PUT("/:userId/level", memberApi.AdjustLevel)
		memberRouter.PUT("/:userId/growth", memberApi.AdjustGrowth)
		memberRouter.PUT("/:userId/status", memberApi.UpdateStatus)
	}
}
