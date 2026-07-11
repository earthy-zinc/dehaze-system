package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

func RegisterAlgorithmRoutes(rg *gin.RouterGroup, algorithmApi *api.AlgorithmApi) {
	algorithmRouterGroup := rg.Group("/algorithms")

	{
		// 读操作 - 无需额外权限
		algorithmRouterGroup.GET("", algorithmApi.GetList)             // 获取算法树形表格
		algorithmRouterGroup.GET("/compare", algorithmApi.Compare)     // 算法对比
		algorithmRouterGroup.GET("/options", algorithmApi.GetOptions)  // 获取模型下拉选项列表
		algorithmRouterGroup.GET("/favorites", algorithmApi.ListFavorites)
		algorithmRouterGroup.GET("/favorites/check", algorithmApi.CheckFavorite)
		algorithmRouterGroup.GET("/:id", algorithmApi.GetById)         // 根据ID获取算法信息

		// 写操作 - 需要权限校验
		algorithmRouterGroup.POST("/:id/favorite", algorithmApi.ToggleFavorite) // 切换收藏
		algorithmRouterGroup.POST("", middleware.Permission("sys:algorithm:add"), algorithmApi.Add)                // 新增算法
		algorithmRouterGroup.PUT("/:id", middleware.Permission("sys:algorithm:edit"), algorithmApi.Update)          // 修改算法
		algorithmRouterGroup.PUT("/:id/status", middleware.Permission("sys:algorithm:edit"), algorithmApi.UpdateStatus) // 更新算法状态
		algorithmRouterGroup.DELETE("", middleware.Permission("sys:algorithm:delete"), algorithmApi.Delete)            // 删除算法
	}
}
