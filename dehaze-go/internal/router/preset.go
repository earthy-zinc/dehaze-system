package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/gin-gonic/gin"
)

func RegisterPresetRoutes(rg *gin.RouterGroup, presetApi *api.SysPresetApi) {
	presetGroup := rg.Group("/presets")

	presetGroup.GET("", presetApi.ListPresets)         // 获取预设列表
	presetGroup.POST("", presetApi.CreatePreset)       // 创建自定义预设
	presetGroup.PUT("/:id", presetApi.UpdatePreset)    // 更新自定义预设
	presetGroup.DELETE("/:id", presetApi.DeletePreset) // 删除自定义预设
}
