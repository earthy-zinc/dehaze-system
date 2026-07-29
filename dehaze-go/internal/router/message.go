package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterMessageRoutes(rg *gin.RouterGroup, messageApi *api.MessageApi) {
	msgRouter := rg.Group("/messages")
	{
		msgRouter.GET("", messageApi.GetPage)
		msgRouter.GET("/unread-count", messageApi.GetUnreadCount)
		msgRouter.GET("/search", messageApi.Search)
		msgRouter.GET("/:id", messageApi.GetDetail)
		msgRouter.PATCH("/_read-all", messageApi.MarkAllRead)
		msgRouter.PATCH("/:id/_read", messageApi.MarkRead)
		msgRouter.DELETE("/:ids", messageApi.Delete)
		msgRouter.POST("/send", messageApi.Send)
	}
}

func RegisterNotificationSettingRoutes(rg *gin.RouterGroup, settingApi *api.NotificationSettingApi) {
	settingRouter := rg.Group("/notification-settings")
	{
		settingRouter.GET("", settingApi.Get)
		settingRouter.PATCH("", settingApi.Update)
	}
}

func RegisterAnnouncementRoutes(rg *gin.RouterGroup, annApi *api.AnnouncementApi) {
	annRouter := rg.Group("/announcements")
	{
		annRouter.GET("/page", annApi.GetPage)
		annRouter.POST("", middleware.Permission("notify:announcement:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), annApi.Create)
		annRouter.GET("/:id", annApi.GetDetail)
		annRouter.PUT("/:id", middleware.Permission("notify:announcement:edit"), annApi.Update)
		annRouter.DELETE("/:id", middleware.Permission("notify:announcement:delete"), annApi.Delete)
		annRouter.POST("/:id/_send", middleware.Permission("notify:announcement:send"), annApi.Send)
		annRouter.PATCH("/:id/_cancel", middleware.Permission("notify:announcement:cancel"), annApi.Cancel)
	}
}

func RegisterMessageTemplateRoutes(rg *gin.RouterGroup, tplApi *api.MessageTemplateApi) {
	tplRouter := rg.Group("/message-templates")
	{
		tplRouter.GET("/page", tplApi.GetPage)
		tplRouter.GET("/:id", tplApi.GetDetail)
		tplRouter.PUT("/:id", middleware.Permission("notify:template:edit"), tplApi.Update)
	}
}
