package router

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/api"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	"github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// TestMemberStaticSubPathRouteOrder 路由顺序守卫：
// `/members/benefit-summary` 与 `/members/trial-status` 都是**静态子路径**，必须先于 `/members/:userId`
// 注册，否则会被 `:userId` 捕获、走参数解析并报「用户ID格式不正确」（这两个端点在集成测试里都曾因此失败）。
//
// 用真实路由表 + 桩服务断言两条路径各自命中自己的 handler：桩 EXPECT 未命中会在用例结束时报错，
// 因此「被 :userId 吞掉」或「误路由到 detail」都会立即失败。
func TestMemberStaticSubPathRouteOrder(t *testing.T) {
	gin.SetMode(gin.TestMode)

	memberSvc := mocks.NewMockIMemberService(t)
	memberSvc.EXPECT().GetBenefitSummary(mock.Anything, int64(7)).
		Return(&vo.MemberBenefitSummaryVO{
			ImageCategory: vo.BenefitImageCategoryVO{
				Remaining: 3,
				Details: []vo.BenefitImageTaskVO{
					{TaskType: "dehaze", Quota: 10, Used: 7, Remaining: 3},
				},
			},
			EvaluateCategory: vo.BenefitEvaluateCategoryVO{Remaining: 5},
		}, nil)
	memberSvc.EXPECT().GetTrialStatus(mock.Anything, int64(7)).
		Return(&vo.MemberTrialStatusVO{ShowTrialEntry: true, TrialDays: 3, TrialCredits: 100}, nil)

	engine := gin.New()
	engine.Use(func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 7})
		c.Next()
	})
	RegisterMemberRoutes(engine.Group("/api/v1"), api.NewMemberApi(memberSvc, nil))

	cases := []struct {
		path   string
		marker string
	}{
		{"/api/v1/members/benefit-summary", "imageCategory"},
		{"/api/v1/members/trial-status", "showTrialEntry"},
	}
	for _, tc := range cases {
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, tc.path, nil))

		body := rec.Body.String()
		require.Contains(t, body, tc.marker, tc.path+" 必须命中自身 handler")
		require.NotContains(t, body, "用户ID格式不正确", tc.path+" 不得被 /:userId 抢先匹配")
	}
}
