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
		// 注意：静态路径（/compare、/options、/favorites 等）
		// 必须在 /:id 之前注册，避免被 /:id 参数路由匹配
		algorithmRouterGroup.GET("", algorithmApi.GetList)            // 获取算法树形表格
		algorithmRouterGroup.GET("/compare", algorithmApi.Compare)    // 算法对比
		algorithmRouterGroup.GET("/options", algorithmApi.GetOptions)  // 获取模型下拉选项列表
		algorithmRouterGroup.GET("/list", algorithmApi.ListAll)        // 获取所有算法扁平列表
		algorithmRouterGroup.GET("/favorites", algorithmApi.ListFavorites)
		algorithmRouterGroup.GET("/favorites/check", algorithmApi.CheckFavorite)

		// /:id 及其子路径
		algorithmRouterGroup.GET("/:id", algorithmApi.GetById)                      // 根据ID获取算法信息
		algorithmRouterGroup.GET("/:id/versions", algorithmApi.GetVersions)         // 获取算法版本历史
		algorithmRouterGroup.GET("/:id/monitor", algorithmApi.GetMonitorData)       // 获取算法监控数据
		algorithmRouterGroup.GET("/:id/monitor/stats", algorithmApi.GetMonitorData) // 获取算法监控统计报表（与 /monitor 返回相同结构）

		// 写操作 - 需要权限校验 + 防重复提交
		algorithmRouterGroup.POST("/:id/favorite", middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), algorithmApi.ToggleFavorite)                     // 切换收藏
		algorithmRouterGroup.POST("", middleware.Permission("sys:algorithm:add"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), algorithmApi.Add) // 新增算法
		algorithmRouterGroup.PUT("/:id", middleware.Permission("sys:algorithm:edit"), algorithmApi.Update)                                                         // 修改算法
		algorithmRouterGroup.PUT("/:id/status", middleware.Permission("sys:algorithm:edit"), algorithmApi.UpdateStatus)                                            // 更新算法状态
		algorithmRouterGroup.DELETE("", middleware.Permission("sys:algorithm:delete"), algorithmApi.Delete)                                                        // 删除算法
	}
}
