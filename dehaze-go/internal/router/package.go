package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterPackageRoutes(rg *gin.RouterGroup, packageApi *api.PackageApi) {
	pkgRouter := rg.Group("/packages")
	{
		pkgRouter.GET("", packageApi.ListOnSale)
		pkgRouter.GET("/page", packageApi.GetPage)
		pkgRouter.GET("/sales/stats", packageApi.GetSalesStats)
		pkgRouter.GET("/calculate-price", packageApi.CalculatePrice)
		pkgRouter.POST("", middleware.Permission("package:add"), packageApi.Add)
		pkgRouter.PUT("/:id", middleware.Permission("package:edit"), packageApi.Update)
		pkgRouter.PUT("/:id/status", middleware.Permission("package:edit"), packageApi.UpdateStatus)
		pkgRouter.GET("/:id", packageApi.GetDetail)
		pkgRouter.GET("/:id/form", packageApi.GetForm)
		pkgRouter.DELETE("/:ids", middleware.Permission("package:delete"), packageApi.DeleteByIds)
	}

	couponRouter := pkgRouter.Group("/coupons")
	{
		couponRouter.GET("/my", packageApi.ListMyCoupons)
		couponRouter.GET("/page", packageApi.GetCouponPage)
		couponRouter.POST("", middleware.Permission("package:coupon:add"), packageApi.AddCoupon)
		couponRouter.POST("/batch", middleware.Permission("package:coupon:distribute"), packageApi.BatchDistributeCoupon)
		couponRouter.PUT("/:id", middleware.Permission("package:coupon:edit"), packageApi.UpdateCoupon)
		couponRouter.POST("/:couponId/receive", packageApi.ReceiveCoupon)
		couponRouter.DELETE("/:ids", middleware.Permission("package:coupon:delete"), packageApi.DeleteCouponsByIds)
	}
}
