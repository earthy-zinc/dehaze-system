package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// RegisterAiSkillRoutes Skills 管理（/api/v1/ai/skills）：
// 读接口登录即可（普通用户仅见启用项），写接口需 ai:skill:manage；路径参数统一 :id。
// upload/test 为 B 类（go-proxy 转发），此处不注册。
func RegisterAiSkillRoutes(rg *gin.RouterGroup, skillApi *api.AiSkillApi) {
	group := rg.Group("/ai/skills")
	manage := middleware.Permission("ai:skill:manage")
	{
		group.GET("", skillApi.ListSkills)
		group.GET("/market", skillApi.ListMarket)
		group.POST("/market", manage, skillApi.ShareToMarket)
		group.GET("/:id", skillApi.GetSkill)
		group.GET("/:id/file", skillApi.GetSkillFile)
		group.POST("", manage, skillApi.CreateSkill)
		group.PUT("/:id", manage, skillApi.UpdateSkill)
		group.PATCH("/:id/status", manage, skillApi.SetSkillStatus)
		group.DELETE("/:id", manage, skillApi.DeleteSkill)
	}
}
