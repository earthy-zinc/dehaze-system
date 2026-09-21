package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/query"
	"github.com/earthyzinc/dehaze-go/internal/model/vo"
	repomocks "github.com/earthyzinc/dehaze-go/internal/repository/mocks"
	algoservice "github.com/earthyzinc/dehaze-go/internal/service/algorithm"
	datasetservice "github.com/earthyzinc/dehaze-go/internal/service/dataset"
	favoriteservice "github.com/earthyzinc/dehaze-go/internal/service/favorite"
	msgservice "github.com/earthyzinc/dehaze-go/internal/service/message"
	pkgsaleservice "github.com/earthyzinc/dehaze-go/internal/service/pkgsale"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/security"
	"github.com/earthyzinc/dehaze-go/pkg/server/gin/middleware"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// pageProbeEngine 注册分页端点探针：伪装 ROOT 越过权限门禁（权限非本文件关注点），
// 且不注入 service——越过参数校验后调用 service 必空指针 panic，"是否 panic"即"是否通过校验"。
// pattern 可带路径参数（部分分页端点在路径上取 :userId/:id）。
func pageProbeEngine(pattern string, handler gin.HandlerFunc) *gin.Engine {
	engine := gin.New()
	engine.Use(middleware.ContextErrorHandler())
	engine.Use(func(c *gin.Context) {
		c.Set("claims", &security.CustomClaims{UserID: 1, Authorities: []string{"ROLE_ROOT"}})
	})
	engine.GET(pattern, handler)
	return engine
}

// assertPageBounds 对一组分页用例断言：拒绝集必须 400+A0400，放行集必须越过校验进入 service 调用。
// requestPath 可自带查询串（如必填的非分页参数），此时用例以 & 追加。
func assertPageBounds(t *testing.T, pattern, requestPath string, handler gin.HandlerFunc, denied, accepted []string) {
	t.Helper()
	separator := "?"
	if strings.Contains(requestPath, "?") {
		separator = "&"
	}
	for _, suffix := range denied {
		t.Run("deny "+suffix, func(t *testing.T) {
			rec := httptest.NewRecorder()
			pageProbeEngine(pattern, handler).ServeHTTP(rec,
				httptest.NewRequest(http.MethodGet, requestPath+separator+suffix, nil))
			assert.Contains(t, rec.Body.String(), common.PARAM_ERROR.Code)
		})
	}
	for _, suffix := range accepted {
		t.Run("accept "+suffix, func(t *testing.T) {
			require.Panics(t, func() {
				pageProbeEngine(pattern, handler).ServeHTTP(httptest.NewRecorder(),
					httptest.NewRequest(http.MethodGet, requestPath+separator+suffix, nil))
			}, "参数应通过校验并进入 service 调用")
		})
	}
}

// TestPaginationSweepRejectsOutOfRangeBounds 全仓分页端点统一口径回归。
//
// python 各端点一律 `pageNum: Query(default=1, ge=1)` / `pageSize: Query(default=N, ge=1, le=100)`。
// 这些端点此前各自手写解析：越界静默回退默认值，导致 `pageSize=500` 在 go 返回 500 行、在 python 报
// A0400（sys_file 更严重，`err == nil` 即采用，连 pageSize=0/负值都放行，负值会让 LIMIT 子句消失）。
// 本用例把"越界必拒 + 边界放行"钉在每个端点上，防止回退到第二套实现。
func TestPaginationSweepRejectsOutOfRangeBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	denied := []string{"pageSize=-1", "pageSize=0", "pageSize=101", "pageNum=0", "pageNum=-1", "pageSize=abc", "pageNum=abc"}
	accepted := []string{"", "pageNum=2&pageSize=50", "pageSize=100"}

	cases := []struct {
		name        string
		pattern     string
		requestPath string
		handler     gin.HandlerFunc
	}{
		// 端点名后的 py 行号＝python 该端点 pageNum/pageSize 的 Query 声明处（默认值由此逐端点核实）
		{"auth.ListLoginLogs(py auth.py:161-162,默认10)", "/probe", "/probe", NewAuthApi(nil).ListLoginLogs},
		{"announcement.GetPage(py announcement.py:20-21,默认10)", "/probe", "/probe", NewAnnouncementApi(nil).GetPage},
		{"dataset.GetDatasetList(py dataset.py:29-30,默认10)", "/probe", "/probe", NewSysDatasetApi(nil, nil).GetDatasetList},
		{"datasetItem.GetDatasetItems(py dataset_item.py:40-41,默认20)", "/probe", "/probe", NewSysDatasetItemApi(nil, nil, nil).GetDatasetItems},
		{"dict.GetDictPage(py dict.py:133-134,默认10)", "/probe", "/probe?typeCode=dict_type", NewSysDictApi(nil, nil).GetDictPage},
		{"dict.GetDictTypePage(py dict.py:37-38,默认10)", "/probe", "/probe", NewSysDictApi(nil, nil).GetDictTypePage},
		{"evaluation.GetMetrics(py evaluation.py:167-168,默认10)", "/probe", "/probe", NewSysEvaluationApi(nil).GetMetrics},
		{"evaluation.ListEvaluationLogs(py evaluation.py:148-149,默认10)", "/probe", "/probe", NewSysEvaluationApi(nil).ListEvaluationLogs},
		{"favorite.GetPage(py favorite.py:25-26,默认20)", "/probe", "/probe", NewFavoriteApi(nil).GetPage},
		{"feedback.ListMyRatings(py feedback.py:46-47,默认10)", "/probe", "/probe", NewFeedbackApi(nil, nil).ListMyRatings},
		{"feedback.ListRatings(py feedback.py:69-70,默认10)", "/probe", "/probe", NewFeedbackApi(nil, nil).ListRatings},
		{"feedback.ListMyFeedback(py feedback.py:167-168,默认10)", "/probe", "/probe", NewFeedbackApi(nil, nil).ListMyFeedback},
		{"feedback.ListFeedback(py feedback.py:180-181,默认10)", "/probe", "/probe", NewFeedbackApi(nil, nil).ListFeedback},
		{"file.GetFilePage(py file.py:139-140,默认10)", "/probe", "/probe", NewSysFileApi(nil).GetFilePage},
		{"inputHistory.ListHistory(py image_input.py:48-49,默认10)", "/probe", "/probe", NewSysInputHistoryApi(nil).ListHistory},
		{"member.GetGrowthLogs(py member.py:40-41,默认10)", "/probe", "/probe", NewMemberApi(nil, nil).GetGrowthLogs},
		{"member.GetPage(py member.py:85-86,默认10)", "/probe", "/probe", NewMemberApi(nil, nil).GetPage},
		{"member.GetMemberGrowthLogs(py member.py:170-171,默认10)", "/probe/:userId", "/probe/1", NewMemberApi(nil, nil).GetMemberGrowthLogs},
		{"member.GetMemberConsumptionRecords(py member.py:196-197,默认10)", "/probe/:userId", "/probe/1", NewMemberApi(nil, nil).GetMemberConsumptionRecords},
		{"member.GetMemberOperationLogs(py member.py:223-224,默认10)", "/probe/:userId", "/probe/1", NewMemberApi(nil, nil).GetMemberOperationLogs},
		{"message.GetPage(py message.py:22-23,默认20)", "/probe", "/probe", NewMessageApi(nil).GetPage},
		{"message.Search(py message.py:45-46,默认20)", "/probe", "/probe", NewMessageApi(nil).Search},
		{"messageTemplate.GetPage(py message_template.py:20-21,默认20)", "/probe", "/probe", NewMessageTemplateApi(nil).GetPage},
		{"order.ListMy(py order.py:90-91,默认10)", "/probe", "/probe", NewOrderApi(nil).ListMy},
		{"order.GetPage(py order.py:107-108,默认10)", "/probe", "/probe", NewOrderApi(nil).GetPage},
		{"order.ListRefunds(py order.py:144-145,默认10)", "/probe", "/probe", NewOrderApi(nil).ListRefunds},
		{"package.GetPage(py package.py:46-47,默认10)", "/probe", "/probe", NewPackageApi(nil, nil).GetPage},
		{"package.GetCouponPage(py package.py:130-131,默认10)", "/probe", "/probe", NewPackageApi(nil, nil).GetCouponPage},
		{"preset.ListPresets(py preset.py:26-27,默认10)", "/probe", "/probe", NewSysPresetApi(nil).ListPresets},
		{"prediction.ListPredictionLogs(py prediction.py:178-179,默认10)", "/probe", "/probe", NewSysPredictionApi(nil, nil).ListPredictionLogs},
		{"promotion.GetPage(py promotion.py:25-26,默认10)", "/probe", "/probe", NewPromotionApi(nil).GetPage},
		{"role.GetRolePage(py role.py:34-36 RolePageQuery:BasePageQuery,默认10)", "/probe", "/probe", NewSysRoleApi(nil).GetRolePage},
		{"task.GetTaskPage(py task.py:66-67,默认10)", "/probe", "/probe", NewSysTaskApi(nil).GetTaskPage},
		{"user.ListPagedUsers(py user.py:25-26,默认10)", "/probe", "/probe", NewSysUserApi(nil).ListPagedUsers},
		// algorithm.GetMonitorStatsReport 不是分页端点（只取 days），由 TestMonitorStatsReportDaysMatchesPython 覆盖
		// algorithm.GetList 不是分页端点：python `GET /algorithms` 无分页参数，由 TestAlgorithmTreeIgnoresPaginationParams 覆盖
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertPageBounds(t, tc.pattern, tc.requestPath, tc.handler, denied, accepted)
		})
	}
}

// TestParsePaginationWithSizeKeepsEndpointDefault 带默认值的变体必须返回端点自报的 defaultSize，
// 而不是恒 10——否则 favorite/message 这类非 10 端点会被"顺手"改成 10。
func TestParsePaginationWithSizeKeepsEndpointDefault(t *testing.T) {
	gin.SetMode(gin.TestMode)
	ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
	ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)

	pageNum, pageSize, ok := parsePaginationWithSize(ctx, 20)
	require.True(t, ok)
	assert.Equal(t, 1, pageNum)
	assert.Equal(t, 20, pageSize)
	assert.Empty(t, ctx.Errors)
}

// pageRecorder 记录 handler 真实透传给 service 的分页值。
// 默认值必须落在"透传值"上：只测 parser 无法发现调用点把 python 的 20 写成了 10。
type pageRecorder struct {
	pageNum, pageSize int
	called            bool
}

type recordingFavoriteService struct {
	favoriteservice.IFavoriteService
	rec *pageRecorder
}

func (s *recordingFavoriteService) GetPage(_ context.Context, _ int64, q *query.FavoritePageQuery) (*vo.PageResult[vo.FavoriteVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

type recordingMessageService struct {
	msgservice.IMessageService
	rec *pageRecorder
}

func (s *recordingMessageService) GetPage(_ context.Context, _ int64, q *query.MessageQuery) (*vo.PageResult[vo.MessageVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

func (s *recordingMessageService) Search(_ context.Context, _ int64, q *query.MessageSearchQuery) (*vo.PageResult[vo.MessageVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

type recordingMessageTemplateService struct {
	msgservice.IMessageTemplateService
	rec *pageRecorder
}

func (s *recordingMessageTemplateService) GetPage(_ context.Context, q *query.MessageTemplateQuery) (*vo.PageResult[vo.MessageTemplateVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

type recordingPackageService struct {
	pkgsaleservice.IPackageService
	rec *pageRecorder
}

func (s *recordingPackageService) GetPage(_ context.Context, q *query.PackagePageQuery) (*vo.PageResult[vo.PackagePageVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

type recordingCouponService struct {
	pkgsaleservice.ICouponService
	rec *pageRecorder
}

func (s *recordingCouponService) GetPage(_ context.Context, q *query.CouponPageQuery) (*vo.PageResult[vo.CouponVO], error) {
	s.rec.pageNum, s.rec.pageSize, s.rec.called = q.PageNum, q.PageSize, true
	return nil, nil
}

// TestPaginationSweepDefaultsMatchPythonRouter 逐端点默认值对 python router 核实（不按"都是 10"推断）：
// favorite.py:26 / message.py:23,46 / message_template.py:21 是 20；package.py:47,131 是 10——
// 后者 go 侧原写 20，属默认值偏离，本次一并对齐。
func TestPaginationSweepDefaultsMatchPythonRouter(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cases := []struct {
		name     string
		wantSize int
		handler  func(rec *pageRecorder) gin.HandlerFunc
	}{
		{"favorite.GetPage(py favorite.py:26=20)", 20, func(rec *pageRecorder) gin.HandlerFunc {
			return NewFavoriteApi(&recordingFavoriteService{rec: rec}).GetPage
		}},
		{"message.GetPage(py message.py:23=20)", 20, func(rec *pageRecorder) gin.HandlerFunc {
			return NewMessageApi(&recordingMessageService{rec: rec}).GetPage
		}},
		{"message.Search(py message.py:46=20)", 20, func(rec *pageRecorder) gin.HandlerFunc {
			return NewMessageApi(&recordingMessageService{rec: rec}).Search
		}},
		{"messageTemplate.GetPage(py message_template.py:21=20)", 20, func(rec *pageRecorder) gin.HandlerFunc {
			return NewMessageTemplateApi(&recordingMessageTemplateService{rec: rec}).GetPage
		}},
		{"package.GetPage(py package.py:47=10)", 10, func(rec *pageRecorder) gin.HandlerFunc {
			return NewPackageApi(&recordingPackageService{rec: rec}, nil).GetPage
		}},
		{"package.GetCouponPage(py package.py:131=10)", 10, func(rec *pageRecorder) gin.HandlerFunc {
			return NewPackageApi(nil, &recordingCouponService{rec: rec}).GetCouponPage
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := &pageRecorder{}
			pageProbeEngine("/probe", tc.handler(rec)).ServeHTTP(httptest.NewRecorder(),
				httptest.NewRequest(http.MethodGet, "/probe", nil))
			require.True(t, rec.called, "未传分页参数时应把默认值透传给 service")
			assert.Equal(t, 1, rec.pageNum)
			assert.Equal(t, tc.wantSize, rec.pageSize)
		})
	}
}

// recordingAlgorithmService 记录统计报表收到的 days。
type recordingAlgorithmService struct {
	algoservice.IAlgorithmService
	days   int
	called bool
}

func (s *recordingAlgorithmService) GetMonitorStatsReport(_ context.Context, _ int64, days int) ([]map[string]interface{}, error) {
	s.days, s.called = days, true
	return nil, nil
}

// TestMonitorStatsReportDaysMatchesPython 与 python `algorithm.py:209 days: int = Query(default=7, ge=1)` 同口径：
// 非整数/非正值 400+A0400（此前静默回退 7——days=0 会在 go 出报表、在 python 报参数错误），不传默认 7，合法值透传。
func TestMonitorStatsReportDaysMatchesPython(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, suffix := range []string{"days=0", "days=-1", "days=abc"} {
		t.Run("deny "+suffix, func(t *testing.T) {
			rec := httptest.NewRecorder()
			pageProbeEngine("/probe/:id", NewAlgorithmApi(nil, nil, nil).GetMonitorStatsReport).ServeHTTP(rec,
				httptest.NewRequest(http.MethodGet, "/probe/1?"+suffix, nil))
			assert.Contains(t, rec.Body.String(), common.PARAM_ERROR.Code)
		})
	}

	cases := []struct {
		query string
		want  int
	}{
		{"", 7},
		{"days=30", 30},
		{"days=1", 1},
	}
	for _, tc := range cases {
		t.Run("accept days="+tc.query, func(t *testing.T) {
			stub := &recordingAlgorithmService{}
			pageProbeEngine("/probe/:id", NewAlgorithmApi(stub, nil, nil).GetMonitorStatsReport).ServeHTTP(
				httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/probe/1?"+tc.query, nil))
			require.True(t, stub.called, "合法 days 应进入 service")
			assert.Equal(t, tc.want, stub.days)
		})
	}
}

// TestPaginationSweepStructBoundEndpoints 结构体绑定型端点的分页口径回归。
//
// 这类端点走 `c.ShouldBindQuery(&bo.XxxQuery)`（内嵌 bo.AiPageQuery）。修复前它们由 service 侧
// 归一化静默钳制：`pageSize=500` 返回 100 行、`pageSize=0` 返回默认行数、`pageNum=0` 返回第 1 页，
// 而 python 侧对应模型全部继承 `BasePageQuery`（ge=1 / le=100）→ A0400；
// `pageNum=abc` 还会被 gin 绑定层兜底成 B0001（应为 A0400）。
//
// 现由 `bo.AiPageQuery` 的 `form:"-"` + handler 内 helper 解析共同保证：分页不参与绑定，
// 越界/非数字一律 A0400。故用例断言与逐参解析型端点完全同口径
// （此前这类形态用 `grep c.Query("pageNum")` 扫不到，是排查盲区）。
func TestPaginationSweepStructBoundEndpoints(t *testing.T) {
	gin.SetMode(gin.TestMode)
	denied := []string{"pageSize=-1", "pageSize=0", "pageSize=101", "pageNum=0", "pageNum=-1", "pageNum=abc", "pageSize=abc"}
	accepted := []string{"", "pageNum=2&pageSize=50", "pageSize=100"}

	cases := []struct {
		name    string
		handler gin.HandlerFunc
	}{
		{"provider.ListProviders(py ProviderPageQuery:BasePageQuery)", NewAiProviderApi(nil).ListProviders},
		{"model.ListModels(py AiModelPageQuery:BasePageQuery)", NewAiModelApi(nil).ListModels},
		{"skill.ListSkills(py SkillPageQuery:BasePageQuery)", NewAiSkillApi(nil).ListSkills},
		{"mcp.ListServers(py McpServerQuery:BasePageQuery)", NewAiMcpApi(nil).ListServers},
		{"mcp.ListCalls(py McpCallQuery:BasePageQuery)", NewAiMcpApi(nil).ListCalls},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertPageBounds(t, "/probe", "/probe", tc.handler, denied, accepted)
		})
	}
}

// TestModelPricePaginationUsesPageSizeNames 模型售价端点的参数名与别的端点不同：
// python `ModelPriceQuery`（schema/ai_model_price.py:63）是 `page`/`size`（ge=1 / ge=1,le=100，默认 20），
// 不吃 pageNum/pageSize。此用例钉住"参数名 + 边界"两件事，防止被套用成通用的 pageNum/pageSize。
//
// 注：该 handler 的 service 是具体类型（`*aiservice.ModelService`）无法打桩，故默认值 20 由调用点
// 字面量（`parsePaginationNamed(c, "page", "size", 20)`）保证，用例只能断言"未传时放行进入 service"。
func TestModelPricePaginationUsesPageSizeNames(t *testing.T) {
	gin.SetMode(gin.TestMode)
	assertPageBounds(t, "/probe/:id", "/probe/1", NewAiModelApi(nil).ListModelPrices,
		[]string{"size=-1", "size=0", "size=101", "page=0", "page=-1", "page=abc", "size=abc"},
		[]string{"", "page=2&size=50", "size=100"},
	)

	// pageNum/pageSize 对该端点是无关参数（python 同样忽略）→ 不参与校验，不应因此报错。
	t.Run("pageNum/pageSize 属无关参数", func(t *testing.T) {
		require.Panics(t, func() {
			pageProbeEngine("/probe/:id", NewAiModelApi(nil).ListModelPrices).ServeHTTP(
				httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/probe/1?pageNum=0&pageSize=500", nil))
		}, "无关参数不应触发校验失败，应正常进入 service")
	})
}

// TestDatasetPaginationDefaultsMatchPython 数据集与其数据项两个端点的分页默认值。
//
// 这两个 handler 的 service 字段是具体类型（无法像 favorite/message 那样用内嵌接口打桩），
// 故改用 mock repository 观察 handler 真正透传下去的分页值——只看调用点字面量证明不了默认值生效。
// python 依据：dataset.py:29-30 默认 1/10；dataset_item.py:40-41 默认 1/20（该端点不吃 BasePageQuery 的 10）。
func TestDatasetPaginationDefaultsMatchPython(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("dataset.GetDatasetList 缺省透传 1/10", func(t *testing.T) {
		datasetRepo := repomocks.NewMockIDatasetRepository(t)
		datasetRepo.EXPECT().FindRootPage(mock.Anything, mock.Anything).
			RunAndReturn(func(_ context.Context, q *query.DatasetQuery) ([]model.SysDataset, int64, error) {
				assert.Equal(t, 1, q.PageNum, "pageNum 缺省应为 1")
				assert.Equal(t, 10, q.PageSize, "pageSize 缺省应为 10")
				return []model.SysDataset{}, 0, nil
			})
		service := datasetservice.NewDatasetService(nil, datasetRepo,
			repomocks.NewMockIDatasetItemRepository(t), repomocks.NewMockIDatasetStatsRepository(t),
			repomocks.NewMockIItemFileRepository(t), repomocks.NewMockIFileRepository(t))

		rec := httptest.NewRecorder()
		pageProbeEngine("/probe", NewSysDatasetApi(service, nil).GetDatasetList).ServeHTTP(rec,
			httptest.NewRequest(http.MethodGet, "/probe", nil))
		assert.NotContains(t, rec.Body.String(), common.PARAM_ERROR.Code)
	})

	t.Run("datasetItem.GetDatasetItems 缺省透传 1/20", func(t *testing.T) {
		itemRepo := repomocks.NewMockIDatasetItemRepository(t)
		itemRepo.EXPECT().FindPage(mock.Anything, int64(0), 1, 20).
			Return([]model.SysDatasetItem{}, int64(0), nil)
		service := datasetservice.NewDatasetItemService(nil, itemRepo, nil, nil, nil, nil, nil)

		rec := httptest.NewRecorder()
		pageProbeEngine("/probe", NewSysDatasetItemApi(service, nil, nil).GetDatasetItems).ServeHTTP(rec,
			httptest.NewRequest(http.MethodGet, "/probe", nil))
		assert.NotContains(t, rec.Body.String(), common.PARAM_ERROR.Code)
	})
}

// TestAlgorithmTreeIgnoresPaginationParams python `GET /algorithms` 只声明 keywords（无分页参数），
// 未知查询参数静默忽略。go 若把分页字段留在查询对象上参与绑定，`pageNum=abc` 会因绑定失败报 A0400，
// 而 python 正常返回树（客户端可触发的分叉）；故这些参数必须一律不影响请求成败。
func TestAlgorithmTreeIgnoresPaginationParams(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, q := range []string{"pageNum=abc", "pageSize=abc", "pageNum=0&pageSize=0", "pageSize=101", "pageNum=-1"} {
		t.Run("ignore "+q, func(t *testing.T) {
			require.Panics(t, func() {
				pageProbeEngine("/probe", NewAlgorithmApi(nil, nil, nil).GetList).ServeHTTP(
					httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/probe?"+q, nil))
			}, "python 未声明的分页参数应被静默忽略并进入 service，而不是报 A0400")
		})
	}
}

// TestMessageListCursorPaginationBounds 会话消息列表已改为 before/limit 游标分页
// （旧 pageNum/pageSize 契约作废，不再走 parsePagination*）：
// before 可选（>=1，仅取 id < before，缺省取最新一页）；limit 默认 50、范围 1..100。
// 越界/非数字一律 A0400；旧的 pageNum/pageSize 作为无关参数被忽略——一旦回退到旧分页分支，
// 拒绝集（before/limit 越界）或"旧参无关"子用例即转红。
func TestMessageListCursorPaginationBounds(t *testing.T) {
	gin.SetMode(gin.TestMode)
	handler := NewAiConversationApi(nil, nil, nil, nil).ListMessages

	assertPageBounds(t, "/probe/:id", "/probe/1", handler,
		[]string{"limit=-1", "limit=0", "limit=101", "before=0", "before=-1", "limit=abc", "before=abc"},
		[]string{"", "before=100&limit=50", "limit=100"},
	)

	t.Run("pageNum/pageSize 属无关参数", func(t *testing.T) {
		require.Panics(t, func() {
			pageProbeEngine("/probe/:id", handler).ServeHTTP(
				httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/probe/1?pageNum=0&pageSize=101", nil))
		}, "游标端点不应再读取 pageNum/pageSize，应正常进入 service")
	})
}
