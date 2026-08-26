package prediction

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	algorepo "github.com/earthyzinc/dehaze-go/internal/repository/algorithm"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	predrepo "github.com/earthyzinc/dehaze-go/internal/repository/pred_log"
	memberservice "github.com/earthyzinc/dehaze-go/internal/service/member"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/algorithm"
	"github.com/earthyzinc/dehaze-go/pkg/config/options"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// newRealMemberService 组装真实 MemberService（真实 MySQL + 真实 repository）：
// 使 PredictionService.refundQuota 转发的 RefundQuota 落到真实 DB，
// 从而断言配额真实恢复（而非 mock 调用序列）。
func newRealMemberService(t *testing.T, db *gorm.DB) memberservice.IMemberService {
	t.Helper()
	return memberservice.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		nil, // cache：不启用 Redis，配额扣减/回补直接走 DB
		nil, // auditLogSvc
		nil, // messageSender
		nil, // lifecycle
	)
}

// newPredictionService 组装 PredictionService：
// repository 真实；algo.Client 指向 httptest 算法服务（外部 HTTP 依赖，用真实协议测）；memberSvc 可注入。
func newPredictionService(t *testing.T, db *gorm.DB, algoServerURL string, memberSvc memberservice.IMemberService) *PredictionService {
	t.Helper()
	client, err := algorithm.NewClient(options.Algorithm{
		ServiceURL: algoServerURL,
		MaxRetry:   0,
		Timeout:    5,
	})
	require.NoError(t, err)
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return NewPredictionService(
		predrepo.NewPredLogRepository(db),
		algorepo.NewAlgorithmRepository(db),
		client,
		nil, // cache
		memberSvc,
		lm,
	)
}

// algoResponse 构造 Python 算法服务的标准信封响应 {"code":"...","msg":"...","data":...}
func algoResponse(code string, data any) string {
	raw, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf(`{"code":%q,"msg":"ok","data":%s}`, code, string(raw))
}

func createPredLog(t *testing.T, db *gorm.DB, userID, algorithmID int64) *model.SysPredLog {
	t.Helper()
	log := &model.SysPredLog{
		BaseModel:   model.BaseModel{CreateBy: userID},
		AlgorithmID: algorithmID,
		OriginMD5:   "abc123abc123abc123abc123abc123ab",
		OriginURL:   "http://test.local/origin.png",
		Status:      model.LogStatusProcessing,
	}
	require.NoError(t, db.Create(log).Error)
	return log
}

func createMember(t *testing.T, db *gorm.DB, userID int64, dehazeUsed int) *model.SysMember {
	t.Helper()
	m := &model.SysMember{
		UserID:               userID,
		LevelCode:            "level_0",
		LevelSource:          "growth",
		MonthlyDehazeQuota:   5,
		MonthlyDehazeUsed:    dehazeUsed,
		MonthlyEvaluateQuota: 5,
		Status:               1,
	}
	require.NoError(t, db.Create(m).Error)
	return m
}

func getMember(t *testing.T, db *gorm.DB, userID int64) model.SysMember {
	t.Helper()
	var m model.SysMember
	require.NoError(t, db.Where("user_id = ?", userID).First(&m).Error)
	return m
}

// TestExecuteAsync_Failure_RefundsQuota 预测失败后回补：
// 算法 HTTP 客户端返回业务失败 → pred_log 置 failed，会员去雾已用配额真实 -1。
func TestExecuteAsync_Failure_RefundsQuota(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900101)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(algoResponse("E50001", nil))) // 业务错误，不可重试
	}))
	t.Cleanup(srv.Close)

	_ = createMember(t, db, userID, 1) // 已用 1 次
	svc := newPredictionService(t, db, srv.URL, newRealMemberService(t, db))
	log := createPredLog(t, db, userID, 10)

	svc.executeAsync(context.Background(), log.ID, 10, "http://test.local/origin.png", "{}", "abc123abc123abc123abc123abc123ab", userID)

	// 预测日志标记失败
	var savedLog model.SysPredLog
	require.NoError(t, db.First(&savedLog, log.ID).Error)
	assert.Equal(t, model.LogStatusFailed, savedLog.Status)
	require.NotNil(t, savedLog.ErrorMessage)

	// 配额真实回补：used 1 -> 0
	member := getMember(t, db, userID)
	assert.Equal(t, 0, member.MonthlyDehazeUsed, "失败后去雾已用配额应回补 1")
}

// TestExecuteAsync_Success_NoRefund 预测成功不应误回补配额。
func TestExecuteAsync_Success_NoRefund(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900102)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(algoResponse("00000", map[string]any{
			"logId": 1, "status": 2, "resultUrl": "http://result.local/out.png", "resultThumbnailUrl": "", "time": 3,
		})))
	}))
	t.Cleanup(srv.Close)

	_ = createMember(t, db, userID, 1)
	svc := newPredictionService(t, db, srv.URL, newRealMemberService(t, db))
	log := createPredLog(t, db, userID, 10)

	svc.executeAsync(context.Background(), log.ID, 10, "http://test.local/origin.png", "{}", "abc123abc123abc123abc123abc123ab", userID)

	var savedLog model.SysPredLog
	require.NoError(t, db.First(&savedLog, log.ID).Error)
	assert.Equal(t, model.LogStatusCompleted, savedLog.Status)

	member := getMember(t, db, userID)
	assert.Equal(t, 1, member.MonthlyDehazeUsed, "成功路径不应回补配额")
}

// TestRefundQuota_RestoresUsed 直接验证 refundQuota 把已用配额真实回补。
func TestRefundQuota_RestoresUsed(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900103)
	_ = createMember(t, db, userID, 2)
	svc := newPredictionService(t, db, "http://127.0.0.1:1", newRealMemberService(t, db)) // 不回补路径不需要算法服务

	svc.refundQuota(context.Background(), userID)

	member := getMember(t, db, userID)
	assert.Equal(t, 1, member.MonthlyDehazeUsed, "回补后已用配额应 -1")
}

// TestRefundQuota_DoubleRefund_NoOp 重复回补的实际语义：
// IncrementQuotaUsed(-1) 带下限保护（used > 0 才更新），used 降到 0 后
// 第二次回补为 no-op，used 恒 ≥0，不会因重复回补减为负值。
func TestRefundQuota_DoubleRefund_NoOp(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900104)
	_ = createMember(t, db, userID, 1)
	svc := newPredictionService(t, db, "http://127.0.0.1:1", newRealMemberService(t, db))

	svc.refundQuota(context.Background(), userID)
	svc.refundQuota(context.Background(), userID) // 第二次回补：used 已为 0，no-op

	member := getMember(t, db, userID)
	assert.Equal(t, 0, member.MonthlyDehazeUsed, "下限保护：重复回补为 no-op，used 不得减为负")
}

// TestRefundQuota_UserMissing_NoOp 回补时用户（会员记录）已不存在：
// IncrementQuotaUsed 更新 0 行且不报错，refundQuota 静默无副作用。
func TestRefundQuota_UserMissing_NoOp(t *testing.T) {
	db := testutil.NewTestDB(t)
	userID := int64(900105) // 不创建会员记录
	svc := newPredictionService(t, db, "http://127.0.0.1:1", newRealMemberService(t, db))

	svc.refundQuota(context.Background(), userID) // 应无 panic / 无错误（只记录 warn 日志）

	var memberCount int64
	require.NoError(t, db.Model(&model.SysMember{}).Where("user_id = ?", userID).Count(&memberCount).Error)
	assert.Equal(t, int64(0), memberCount, "不存在的用户不应产生任何会员副作用")
}
