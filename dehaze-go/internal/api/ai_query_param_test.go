package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// probeEngine 构造只含参数校验关注点的探针路由：伪装 ROOT 以越过权限门禁（权限非本文件关注点），
// 且不注入 service——越过参数校验后必然在调用 service 时空指针 panic，"是否 panic"即"是否通过校验"。
func probeEngine(handler gin.HandlerFunc) *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.Use(func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 1, Authorities: []string{"ROLE_ROOT"}})
	})
	engine.GET("/probe", handler)
	return engine
}

// assertQueryParamBounds 对一组用例断言：拒绝集必须返回 A0400，放行集必须越过校验（panic 判定）。
func assertQueryParamBounds(t *testing.T, handler gin.HandlerFunc, denied, accepted []string) {
	t.Helper()
	for _, query := range denied {
		t.Run("deny "+query, func(t *testing.T) {
			rec := httptest.NewRecorder()
			probeEngine(handler).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/probe?"+query, nil))
			assert.Contains(t, rec.Body.String(), common.PARAM_ERROR.Code)
		})
	}
	for _, query := range accepted {
		t.Run("accept "+query, func(t *testing.T) {
			require.Panics(t, func() {
				probeEngine(handler).ServeHTTP(httptest.NewRecorder(),
					httptest.NewRequest(http.MethodGet, "/probe?"+query, nil))
			}, "参数应通过校验并进入 service 调用")
		})
	}
}

// TestQueryTimeRangeRejectsMalformed 时间过滤参数与 python datetime 类型参数同口径：
// 未传表示不过滤；非空但格式非法必须 A0400，不得静默丢掉过滤条件（静默丢弃会返回未过滤结果，
// 与 python 报错形成可触发的行为分叉）。
func TestQueryTimeRangeRejectsMalformed(t *testing.T) {
	gin.SetMode(gin.TestMode)
	valid := []string{"2026-09-17T10:00:00+08:00", "2026-09-17T10:00:00", "2026-09-17 10:00:00", "2026-09-17"}
	for _, value := range valid {
		t.Run("accept "+value, func(t *testing.T) {
			values := url.Values{"startTime": {value}, "endTime": {value}}
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Request = httptest.NewRequest(http.MethodGet, "/?"+values.Encode(), nil)

			start, end, ok := queryTimeRange(ctx, "startTime", "endTime")
			require.True(t, ok)
			require.NotNil(t, start)
			require.NotNil(t, end)
			assert.Empty(t, ctx.Errors)
		})
	}

	t.Run("accept absent", func(t *testing.T) {
		ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
		ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)

		start, end, ok := queryTimeRange(ctx, "startTime", "endTime")
		require.True(t, ok)
		assert.Nil(t, start)
		assert.Nil(t, end)
	})

	denied := []string{
		"startTime=not-a-time",
		"startTime=2026-13-45",
		"endTime=not-a-time",
		"startTime=2026-09-17&endTime=2026/09/17",
	}
	for _, query := range denied {
		t.Run("deny "+query, func(t *testing.T) {
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Request = httptest.NewRequest(http.MethodGet, "/?"+query, nil)

			_, _, ok := queryTimeRange(ctx, "startTime", "endTime")
			require.False(t, ok)
			require.Len(t, ctx.Errors, 1)
			bizErr, isBiz := common.AsBizError(ctx.Errors[0].Err)
			require.True(t, isBiz)
			assert.Equal(t, common.PARAM_ERROR.Code, bizErr.Code().Code)
		})
	}
}

// TestEvalTrendsLimitBoundary 评测趋势 limit 与 python Query(default=100, ge=1, le=500) 同口径：
// 越界报 A0400；边界值 500 必须放行（防过度收紧）。
func TestEvalTrendsLimitBoundary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	assertQueryParamBounds(t,
		NewAiEvalApi(nil, nil).EvalTrends,
		[]string{"limit=0", "limit=501", "limit=abc", "limit=-1", "startTime=not-a-time"},
		[]string{"limit=1", "limit=500", ""},
	)
}

// TestRangedQueryParamBounds 标量过滤参数区间与 python Query 的 ge/le 逐一对齐：
// 越界或非数字返回 A0400，边界值与缺省必须放行——两侧都断言，防止用"过度收紧"换通过。
func TestRangedQueryParamBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("agent 列表 status 区间 [0,1]", func(t *testing.T) {
		assertQueryParamBounds(t,
			NewAiAgentApi(nil, nil, nil).ListAgents,
			[]string{"status=2", "status=-1", "status=abc"},
			[]string{"status=0", "status=1", ""},
		)
	})

	t.Run("A2A 端点列表 status 区间 [0,1]", func(t *testing.T) {
		assertQueryParamBounds(t,
			NewAiAgentApi(nil, nil, nil).ListEndpoints,
			[]string{"status=2", "status=-1", "status=abc"},
			[]string{"status=0", "status=1", ""},
		)
	})

	t.Run("评测复核 status 区间 [1,2]", func(t *testing.T) {
		assertQueryParamBounds(t,
			NewAiEvalApi(nil, nil).EvalReviews,
			[]string{"status=0", "status=3", "status=abc"},
			[]string{"status=1", "status=2", ""},
		)
	})

	// 记忆检索 limit：python 声明为裸 int 且**无 le 上界**，故此处只做类型校验 + 下界保护
	// （GORM 对负 Limit 视为"取消 LIMIT"会返回全量，而 python 的 SQLAlchemy 直接报错，故负值必须拒绝）。
	// 大值必须放行——不得"顺手"加上界，否则比 python 更严。
	t.Run("记忆检索 limit 正整数下界、无上界", func(t *testing.T) {
		assertQueryParamBounds(t,
			NewAiMemoryApi(nil).SearchMemories,
			[]string{"keyword=k&limit=0", "keyword=k&limit=-1", "keyword=k&limit=abc"},
			[]string{"keyword=k&limit=1", "keyword=k&limit=100000", "keyword=k"},
		)
	})
}
