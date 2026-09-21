package api

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestParsePaginationMatchesPythonBasePageQuery 分页参数与 python BasePageQuery 同口径
// （pageNum ge=1、pageSize ge=1,le=100）：未传取默认 1/10，越界或非数字一律 A0400。
//
// 必须"报错"而非静默钳制：静默钳制会让 pageSize=500 在 go 返回 500 行、在 python 报错，
// 形成客户端可触发的行为分叉。
func TestParsePaginationMatchesPythonBasePageQuery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cases := []struct {
		query             string
		wantNum, wantSize int
		wantOK            bool
	}{
		{query: "", wantNum: 1, wantSize: 10, wantOK: true},
		{query: "pageNum=3&pageSize=20", wantNum: 3, wantSize: 20, wantOK: true},
		{query: "pageNum=1&pageSize=1", wantNum: 1, wantSize: 1, wantOK: true},
		{query: "pageNum=1&pageSize=100", wantNum: 1, wantSize: 100, wantOK: true},
		{query: "pageNum=1&pageSize=101", wantOK: false},
		{query: "pageNum=1&pageSize=500", wantOK: false},
		{query: "pageNum=0&pageSize=10", wantOK: false},
		{query: "pageNum=-1&pageSize=10", wantOK: false},
		{query: "pageNum=1&pageSize=0", wantOK: false},
		{query: "pageNum=abc&pageSize=10", wantOK: false},
		{query: "pageNum=1&pageSize=abc", wantOK: false},
	}
	for _, tc := range cases {
		t.Run(tc.query, func(t *testing.T) {
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Request = httptest.NewRequest(http.MethodGet, "/?"+tc.query, nil)

			pageNum, pageSize, ok := parsePagination(ctx)
			require.Equal(t, tc.wantOK, ok)
			if !tc.wantOK {
				require.Len(t, ctx.Errors, 1, "越界分页必须写回业务错误")
				bizErr, isBiz := common.AsBizError(ctx.Errors[0].Err)
				require.True(t, isBiz)
				assert.Equal(t, common.PARAM_ERROR.Code, bizErr.Code().Code)
				return
			}
			assert.Equal(t, tc.wantNum, pageNum)
			assert.Equal(t, tc.wantSize, pageSize)
			assert.Empty(t, ctx.Errors)
		})
	}
}

// TestParsePaginationOutOfRangeRendersA0400 端到端确认越界分页对客户端就是 A0400
// （而不是"校验代码存在但没生效"）：经真实错误渲染中间件断言响应报文里的错误码。
func TestParsePaginationOutOfRangeRendersA0400(t *testing.T) {
	gin.SetMode(gin.TestMode)
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.GET("/paged", func(c *gin.Context) {
		pageNum, pageSize, ok := parsePagination(c)
		if !ok {
			return
		}
		common.OkWithData(gin.H{"pageNum": pageNum, "pageSize": pageSize}, c)
	})

	denied := httptest.NewRecorder()
	engine.ServeHTTP(denied, httptest.NewRequest(http.MethodGet, "/paged?pageNum=1&pageSize=101", nil))
	assert.Contains(t, denied.Body.String(), common.PARAM_ERROR.Code)

	allowed := httptest.NewRecorder()
	engine.ServeHTTP(allowed, httptest.NewRequest(http.MethodGet, "/paged?pageNum=1&pageSize=100", nil))
	assert.NotContains(t, allowed.Body.String(), common.PARAM_ERROR.Code)
	assert.Contains(t, allowed.Body.String(), `"pageSize":100`)
}

// TestPromotionPageQueryRejectsOutOfRangeBounds 促销列表必须走全局分页口径，
// 而不是各自手写静默回退：python `PromotionQuery(BasePageQuery)` 对 pageNum=0/-1、
// pageSize=0/-1/101 一律 A0400，go 若静默取默认值就会"返回结果且成功"，
// 与 python 形成客户端可触发的行为分叉（这正是三端 pageSize 上界统一为 le=100 的 go 侧收口）。
//
// 用 nil service 是刻意的：拒绝集一旦未拦住就会调用 service 并 panic，断言即失败。
func TestPromotionPageQueryRejectsOutOfRangeBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	assertQueryParamBounds(t, NewPromotionApi(nil).GetPage,
		[]string{
			"pageNum=1&pageSize=-1",
			"pageNum=1&pageSize=0",
			"pageNum=1&pageSize=101",
			"pageNum=0&pageSize=10",
			"pageNum=-1&pageSize=10",
			"pageNum=abc&pageSize=10",
			"pageNum=1&pageSize=abc",
		},
		[]string{"", "pageNum=2&pageSize=50", "pageNum=1&pageSize=100"},
	)
}
