package router

import (
	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
)

var importExportModules = []string{"user", "role", "dept", "menu", "dict", "dataset", "algorithm"}

// modulePathSegment 返回模块对应的 URL 路径段。
// 算法模块 CRUD 使用复数 /algorithms，导入导出路径保持一致。
func modulePathSegment(module string) string {
	if module == "algorithm" {
		return "algorithms"
	}
	return module
}

func RegisterImportExportRoutes(rg *gin.RouterGroup, importExportApi *api.ImportExportApi) {
	for _, m := range importExportModules {
		module := m
		path := modulePathSegment(module)
		rg.GET("/"+path+"/_export", setModule(module), modulePermission("export"), importExportApi.Export)
		rg.POST("/"+path+"/_export", setModule(module), modulePermission("export"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 3}), importExportApi.ExportPost)
		rg.POST("/"+path+"/_import", setModule(module), modulePermission("import"), middleware.AntiRepeat(middleware.AntiRepeatConfig{Expire: 5}), importExportApi.Import)
		rg.GET("/"+path+"/template", setModule(module), modulePermission("import"), importExportApi.DownloadTemplate)
	}
}

func setModule(module string) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Set("importExportModule", module)
		c.Next()
	}
}

func modulePermission(action string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if security.IsRoot(c) {
			c.Next()
			return
		}
		module, _ := c.Get("importExportModule")
		moduleStr, _ := module.(string)
		if moduleStr == "" {
			moduleStr = c.Param("module")
		}
		hasPerm, err := security.HasPermission(c, "sys:"+moduleStr+":"+action)
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
