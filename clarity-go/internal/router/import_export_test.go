package router

import (
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// TestImportExportRoutesRegisterStaticAndDynamic 导入导出路由必须**静态 + 动态**两套都在：
// 静态用于避开模块自身的 `/{module}/:id` CRUD 路由（如 `/dict/template` 否则被 `:id` 抢走），
// 动态 `/:module/...` 保证未知模块也能命中路由、再由 service registry 判支持性（A0710）。
func TestImportExportRoutesRegisterStaticAndDynamic(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	v1 := engine.Group("/api/v1")
	require.NotPanics(t, func() {
		RegisterImportExportRoutes(v1, nil)
	})

	registered := make(map[string]bool)
	for _, route := range engine.Routes() {
		registered[route.Method+" "+route.Path] = true
	}
	expected := []string{
		http.MethodGet + " /api/v1/user/_export",
		http.MethodGet + " /api/v1/dict/template",
		http.MethodPost + " /api/v1/role/_import",
		http.MethodGet + " /api/v1/:module/_export",
		http.MethodPost + " /api/v1/:module/_export",
		http.MethodPost + " /api/v1/:module/_import",
		http.MethodGet + " /api/v1/:module/template",
	}
	for _, path := range expected {
		require.True(t, registered[path], "路由未注册: %s", path)
	}
}

// TestWithModuleFillsModuleForStaticRoutes 静态导入导出路由必须补齐模块名：
// handler 的 getModule 先读 ctx 值 `importExportModule`、再读 c.Param("module")，
// modulePermission 读 c.Param("module") —— 静态路由两者都没有，
// 缺 withModule 时全部静态模块报 A0400「模块名不能为空」（run7 实测 30 处）。
func TestWithModuleFillsModuleForStaticRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := &fakeResponseWriter{}
	c, _ := gin.CreateTestContext(rec)
	c.Params = gin.Params{{Key: "id", Value: "template"}} // 模拟静态匹配：只有模块自身的 :id

	withModule("dict")(c)

	require.Equal(t, "dict", c.GetString("importExportModule"), "getModule 的 ctx 值路径必须可解析")
	require.Equal(t, "dict", c.Param("module"), "modulePermission 的 Param 路径必须可解析")
}

type fakeResponseWriter struct{ header http.Header }

func (f *fakeResponseWriter) Header() http.Header {
	if f.header == nil {
		f.header = http.Header{}
	}
	return f.header
}
func (f *fakeResponseWriter) Write(b []byte) (int, error) { return len(b), nil }
func (f *fakeResponseWriter) WriteHeader(int)             {}
