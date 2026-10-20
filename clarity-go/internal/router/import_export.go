package router

import (
	"github.com/earthyzinc/clarity-go/internal/api"
	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/security"
	"github.com/earthyzinc/clarity-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

// supportedImportExportModules 支持导入导出的业务模块（与 python 侧 handler registry 一致）。
// 路径段即模块名本身（python `GET /api/v1/{module}/_export`），不做单复数映射。
var supportedImportExportModules = []string{"user", "role", "dept", "menu", "dict", "dataset"}

// RegisterImportExportRoutes 通用导入导出路由。
//
// 路径为 `/{module}/_export|_import|template`，模块段即模块名本身；模块是否受支持由 service 层
// 的 handler registry 判定（不支持 → A0710），未知模块靠**动态段兜底**命中路由（python 对任意模块名
// 都命中路由，再由 registry 判支持性）。
//
// 两级注册的原因：部分模块自身有 `/{module}/:id` 这类 CRUD 路由（如 dict），此时只注册动态
// `/:module/template` 会被该组的 `:id` 抢先匹配（`/dict/template` → id="template" → 404）。
// 故先按支持列表注册**静态**路径（本函数须在其它模块路由之前调用，保证静态节点先建、gin 静态优先），
// 再注册动态段兜底。
func RegisterImportExportRoutes(rg *gin.RouterGroup, importExportApi *api.ImportExportApi) {
	for _, module := range supportedImportExportModules {
		rg.GET("/"+module+"/_export", withModule(module), modulePermission("export"), importExportApi.Export)
		// 导出为只读查询（POST 仅因筛选条件复杂），不挂防重；导入为写操作，挂防重
		rg.POST("/"+module+"/_export", withModule(module), modulePermission("export"), importExportApi.ExportPost)
		rg.POST("/"+module+"/_import", withModule(module), modulePermission("import"),
			middleware.DefaultAntiRepeat(), importExportApi.Import)
		rg.GET("/"+module+"/template", withModule(module), modulePermission("import"), importExportApi.DownloadTemplate)
	}

	// 动态兜底段不挂防重：防重白名单只覆盖上方的静态模块集合（与 python/java 白名单一致）
	rg.GET("/:module/_export", modulePermission("export"), importExportApi.Export)
	rg.POST("/:module/_export", modulePermission("export"), importExportApi.ExportPost)
	rg.POST("/:module/_import", modulePermission("import"), importExportApi.Import)
	rg.GET("/:module/template", modulePermission("import"), importExportApi.DownloadTemplate)
}

// withModule 为**静态注册**的导入导出路由补齐模块信息。
// 静态路径（如 /user/_export、/dict/template）没有 `:module` 段，而 handler 的 getModule 与
// modulePermission 都按模块名解析 → 缺这一层会取到空模块名，全部静态模块报 A0400「模块名不能为空」。
func withModule(module string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Set("importExportModule", module)
		c.Params = append(c.Params, gin.Param{Key: "module", Value: module})
		c.Next()
	}
}

// modulePermission 通用导入导出的模块级权限：`sys:{module}:{action}`。
func modulePermission(action string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if security.IsRoot(c) {
			c.Next()
			return
		}
		module := c.Param("module")
		hasPerm, err := security.HasPermission(c, "sys:"+module+":"+action)
		if err != nil {
			_ = c.Error(common.WrapBizError(common.AUTHORIZED_ERROR, "权限校验失败", err))
			c.Abort()
			return
		}
		if !hasPerm {
			_ = c.Error(common.NewBizError(common.AUTHORIZED_ERROR, "权限不足"))
			c.Abort()
			return
		}
		c.Next()
	}
}
