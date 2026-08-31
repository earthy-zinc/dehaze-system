// 资金/权益链路测试：CheckAndDeductQuota / RefundQuota。
//
// 注：本测试使用外部测试包 member_test —— internal/service/mocks 生成的
// MockIMemberService 反向引用 member 包，若本测试声明为 package member 并
// import mocks 会构成导入环；外部测试包规避该环，且被测方法/常量均为导出
// 标识符，不受影响。
//
// 数据层一律使用真实 MySQL 测试库（testutil.NewTestDB / NewPoolTestDB），
// 配额扣减/回补断言真实余额与用量变化，不 mock repository；缓存依赖用
// MockICache。
package member_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	memberrepo "github.com/earthyzinc/dehaze-go/internal/repository/member"
	"github.com/earthyzinc/dehaze-go/internal/service/member"
	servicemocks "github.com/earthyzinc/dehaze-go/internal/service/mocks"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/cache/types"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	"github.com/earthyzinc/dehaze-go/pkg/lifecycle"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// resetMemberTables 清空会员表，规避 config/sql 种子会员（user_id 4~8）对
// 配额/用量断言的干扰。测试在独立事务内执行，回滚后种子数据自动恢复。
func resetMemberTables(t *testing.T, db *gorm.DB) {
	t.Helper()
	require.NoError(t, db.Exec("DELETE FROM sys_member").Error)
}

// newQuotaMember 构造会员行：默认 level_0/正常状态/未删除，配额字段显式注入。
func newQuotaMember(userID int64, dehazeQuota, dehazeUsed, evaluateQuota, evaluateUsed int, status int8) *model.SysMember {
	return &model.SysMember{
		UserID:               userID,
		LevelCode:            "level_0",
		LevelSource:          "growth",
		GrowthValue:          0,
		TotalConsumption:     0,
		Status:               status,
		MonthlyDehazeQuota:   dehazeQuota,
		MonthlyDehazeUsed:    dehazeUsed,
		MonthlyEvaluateQuota: evaluateQuota,
		MonthlyEvaluateUsed:  evaluateUsed,
		Deleted:              0,
	}
}

func mustCreateMember(t *testing.T, db *gorm.DB, m *model.SysMember) {
	t.Helper()
	require.NoError(t, db.Create(m).Error)
}

func getMember(t *testing.T, db *gorm.DB, userID int64) *model.SysMember {
	t.Helper()
	got, err := memberrepo.NewMemberRepository(db).FindByUserID(context.Background(), userID)
	require.NoError(t, err)
	require.NotNil(t, got)
	return got
}

func memberUsed(t *testing.T, db *gorm.DB, userID int64, quotaType member.QuotaType) int {
	t.Helper()
	column := "monthly_dehaze_used"
	if quotaType == member.QuotaTypeEvaluate {
		column = "monthly_evaluate_used"
	}
	var used int
	require.NoError(t, db.Table("sys_member").Where("user_id = ? AND deleted = 0", userID).Pluck(column, &used).Error)
	return used
}

// newQuotaService 按实际构造注入组装：repository 用真实 MySQL 测试库（NewTestDB），
// cache 用 MockICache 或 nil（nil 走同步 DB 落库路径，便于确定性断言），
// audit/message 依赖按需传 nil。lifecycle 必须注入：缓存路径扣减成功会
// s.lifecycle.Go 异步落库，nil 会 panic。
func newQuotaService(t *testing.T, db *gorm.DB, cache types.ICache) *member.MemberService {
	t.Helper()
	lm := lifecycle.NewManager()
	t.Cleanup(func() { _ = lm.Shutdown(2 * time.Second) })
	return member.NewMemberService(
		db,
		memberrepo.NewMemberRepository(db),
		memberrepo.NewMemberBenefitRepository(db),
		memberrepo.NewMemberGrowthLogRepository(db),
		memberrepo.NewMemberSignInRepository(db),
		cache,
		nil, // auditLogSvc
		nil, // messageSender
		lm,
		nil, // dictSvc
	)
}

// waitForQuotaUsed 轮询等待配额用量达到期望值（缓存路径扣减为异步落库）。
func waitForQuotaUsed(t *testing.T, db *gorm.DB, userID int64, quotaType member.QuotaType, want int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for {
		if memberUsed(t, db, userID, quotaType) == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("等待配额用量变化超时: want %d", want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func assertBizCode(t *testing.T, err error, want *common.ResultCode) {
	t.Helper()
	require.Error(t, err)
	bizErr, ok := common.AsBizError(err)
	require.True(t, ok, "期望业务错误 %v，实际: %v", want, err)
	assert.Equal(t, want, bizErr.Code())
}

// ============ CheckAndDeductQuota：正常扣减（DB 同步路径，cache=nil） ============

func TestCheckAndDeductQuota_DBPath_Success(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910001)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 10, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))

	m := getMember(t, db, userID)
	assert.Equal(t, 1, m.MonthlyDehazeUsed, "扣减后已用量应 +1")
	assert.Equal(t, 10, m.MonthlyDehazeQuota, "总配额不变")
	assert.Equal(t, 0, m.MonthlyEvaluateUsed, "去雾扣减不得影响评估用量")
}

func TestCheckAndDeductQuota_DBPath_EvaluateQuotaRouting(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910002)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 20, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeEvaluate))

	m := getMember(t, db, userID)
	assert.Equal(t, 1, m.MonthlyEvaluateUsed, "评估用量应 +1")
	assert.Equal(t, 0, m.MonthlyDehazeUsed, "评估扣减不得影响去雾用量")
}

// 连续扣减多次，断言用量逐次精确 +1、总配额不变（int 精确性，无精度漂移）。
func TestCheckAndDeductQuota_RepeatedDeductions_IntegerExact(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910003)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 0, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	for i := 1; i <= 5; i++ {
		require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))
		assert.Equal(t, i, memberUsed(t, db, userID, member.QuotaTypeDehaze), "第 %d 次扣减后用量应精确为 %d", i, i)
	}
	m := getMember(t, db, userID)
	assert.Equal(t, 5, m.MonthlyDehazeUsed)
	assert.Equal(t, 5, m.MonthlyDehazeQuota)
}

// ============ CheckAndDeductQuota：边界 ============

// 余额恰好等于扣减值（还剩最后 1 次），应扣减成功，用量封顶。
func TestCheckAndDeductQuota_DBPath_LastUnitExactBoundary(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910004)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 4, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 5, memberUsed(t, db, userID, member.QuotaTypeDehaze), "最后 1 次扣减应成功")
}

// 余额已用完（used == quota），返回配额已用尽且余额不变。
func TestCheckAndDeductQuota_DBPath_ExhaustedWhenUsedEqualsQuota(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910005)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 5, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	err := svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
	assertBizCode(t, err, common.QUOTA_EXCEEDED)
	assert.Equal(t, 5, memberUsed(t, db, userID, member.QuotaTypeDehaze), "扣减失败不得改变用量")
}

// 0 额度直接拒绝。
func TestCheckAndDeductQuota_DBPath_ZeroQuota(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910006)
	mustCreateMember(t, db, newQuotaMember(userID, 0, 0, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	err := svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
	assertBizCode(t, err, common.QUOTA_EXCEEDED)
}

// ============ CheckAndDeductQuota：会员前置校验 ============

func TestCheckAndDeductQuota_MemberNotFound(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)

	svc := newQuotaService(t, db, nil)
	err := svc.CheckAndDeductQuota(context.Background(), 919999, member.QuotaTypeDehaze)
	assertBizCode(t, err, common.MEMBER_NOT_FOUND)
}

func TestCheckAndDeductQuota_MemberFrozen(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910007)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 0, 0, 1))
	// status 列带 default:1 标签，gorm 插入零值(0)时会省略该列使 DB 落默认值 1，
	// 故须显式 UPDATE 置 0 以构造冻结态。
	require.NoError(t, db.Model(&model.SysMember{}).Where("user_id = ?", userID).Update("status", 0).Error)

	svc := newQuotaService(t, db, nil)
	err := svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
	assertBizCode(t, err, common.MEMBER_FROZEN)
	assert.Equal(t, 0, memberUsed(t, db, userID, member.QuotaTypeDehaze), "冻结会员扣减失败不得改变用量")
}

// ============ CheckAndDeductQuota：缓存路径（MockICache） ============

// 缓存计数器未命中 → 按 DB 余额初始化计数器 → DECR 成功 → 异步落库。
// 异步落库 goroutine 与断言轮询须走独立连接，故用 NewPoolTestDB（多连接），
// 避免 NewTestDB 单事务连接上并发触发 MySQL 协议错乱。
func TestCheckAndDeductQuota_CachePath_Success(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(910008)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Get(mock.Anything, counterKey).Return("", nil).Once()
	cache.EXPECT().Set(mock.Anything, counterKey, int64(10), mock.Anything).Return(nil).Once()
	cache.EXPECT().Decr(mock.Anything, counterKey).Return(int64(9), nil).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))

	// 缓存路径落库为异步（lifecycle.Go），轮询等待最终一致。
	waitForQuotaUsed(t, db, userID, member.QuotaTypeDehaze, 1)
}

// 缓存计数器命中且剩余 1，DECR 返回 0（恰好扣完），仍算成功。
// 同 CachePath_Success，异步落库用 pool 模式。
func TestCheckAndDeductQuota_CachePath_DecrReturnsZero_LastUnit(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(910009)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Get(mock.Anything, counterKey).Return("1", nil).Once()
	cache.EXPECT().Decr(mock.Anything, counterKey).Return(int64(0), nil).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))
	waitForQuotaUsed(t, db, userID, member.QuotaTypeDehaze, 1)
}

// 缓存 DECR 后变负（计数器已透支）：回滚 Incr 并返回配额已用尽，DB 用量不变。
func TestCheckAndDeductQuota_CachePath_CounterNegative_RollsBack(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910010)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 0, 0, 0, 1))

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Get(mock.Anything, counterKey).Return("1", nil).Once()
	cache.EXPECT().Decr(mock.Anything, counterKey).Return(int64(-1), nil).Once()
	cache.EXPECT().Incr(mock.Anything, counterKey).Return(int64(0), nil).Once()

	svc := newQuotaService(t, db, cache)
	err := svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
	assertBizCode(t, err, common.QUOTA_EXCEEDED)
	assert.Equal(t, 0, memberUsed(t, db, userID, member.QuotaTypeDehaze), "透支回滚后 DB 用量不得变化")
}

// 缓存 DECR 失败（Redis 异常）→ 降级为同步 DB 权威扣减，仍应成功且落库确定；
// 且 DB 扣减成功后缓存计数器须写入 DB 精确剩余值（quota-newUsed=4）完成对齐。
func TestCheckAndDeductQuota_CachePath_DecrError_FallsBackToDB(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910011)
	mustCreateMember(t, db, newQuotaMember(userID, 5, 0, 0, 0, 1))

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Get(mock.Anything, counterKey).Return("", nil).Once()
	cache.EXPECT().Set(mock.Anything, counterKey, int64(5), mock.Anything).Return(nil).Once()
	cache.EXPECT().Decr(mock.Anything, counterKey).Return(int64(0), assert.AnError).Once()
	// DB 扣减成功（used 0→1）后，缓存对齐为 DB 权威剩余值 4。
	cache.EXPECT().Set(mock.Anything, counterKey, int64(4), mock.Anything).Return(nil).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))
	// DB 降级路径为同步，直接断言。
	assert.Equal(t, 1, memberUsed(t, db, userID, member.QuotaTypeDehaze))
}

// ============ RefundQuota：失败回补 ============

func TestRefundQuota_Success_DBPath(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910012)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 3, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 2, memberUsed(t, db, userID, member.QuotaTypeDehaze), "回补 1 次后用量应 -1")
}

func TestRefundQuota_Success_CachePath(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910013)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 3, 0, 0, 1))

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Incr(mock.Anything, counterKey).Return(int64(8), nil).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 2, memberUsed(t, db, userID, member.QuotaTypeDehaze), "缓存回补 + DB 落库同步，用量应 -1")
}

// 缓存 Incr 失败只告警，DB 回补仍必须执行（账目以 DB 为准）。
func TestRefundQuota_CacheIncrError_StillRefundsDB(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910014)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 3, 0, 0, 1))

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	cache.EXPECT().Incr(mock.Anything, counterKey).Return(int64(0), assert.AnError).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 2, memberUsed(t, db, userID, member.QuotaTypeDehaze), "缓存失败不影响 DB 回补")
}

// 重复回补：IncrementQuotaUsed(-1) 带下限保护（used > 0 才更新），
// used 降到 0 后再次回补为 no-op，断言 used 恒 ≥0 且无错误。
func TestRefundQuota_Repeated_SecondRefundNoOp(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910015)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 1, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 0, memberUsed(t, db, userID, member.QuotaTypeDehaze), "首次回补 used 1→0")
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	// 下限保护：used 已为 0，第二次回补 no-op，不再减为负。
	assert.Equal(t, 0, memberUsed(t, db, userID, member.QuotaTypeDehaze))
}

// ============ 并发扣减（pool 模式：多连接真实行锁竞争） ============

// 并发扣减且配额恰等于并发数：全部成功，最终用量精确 = 成功数 × 单价，
// 无超扣/少扣，且无丢失更新（UPDATE 行级原子性）。
func TestCheckAndDeductQuota_Concurrent_NoLostUpdate(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(920001)
	mustCreateMember(t, db, newQuotaMember(userID, 8, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	const workers = 8
	svc := newQuotaService(t, db, nil)

	var wg sync.WaitGroup
	errs := make([]error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			errs[idx] = svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
		}(i)
	}
	wg.Wait()

	success := 0
	for _, err := range errs {
		if err == nil {
			success++
		}
	}

	finalUsed := memberUsed(t, db, userID, member.QuotaTypeDehaze)
	assert.Equal(t, workers, success, "配额=并发数时全部应成功")
	assert.Equal(t, workers, finalUsed, "最终用量应精确 = 成功数 × 1，无丢失更新")
	assert.Equal(t, success, finalUsed, "账目守恒：用量 = 成功扣减数")
}

// 并发扣减但配额小于并发数：行级条件更新（WHERE used < quota）保证无超扣，
// 成功数不得越过配额；账目守恒（用量 == 成功扣减数）。
func TestCheckAndDeductQuota_Concurrent_QuotaLessThanWorkers_NoOversell(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(920002)
	mustCreateMember(t, db, newQuotaMember(userID, 4, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	const workers = 8
	svc := newQuotaService(t, db, nil)

	var wg sync.WaitGroup
	errs := make([]error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			errs[idx] = svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze)
		}(i)
	}
	wg.Wait()

	success := 0
	for _, err := range errs {
		if err == nil {
			success++
		}
	}

	finalUsed := memberUsed(t, db, userID, member.QuotaTypeDehaze)
	assert.GreaterOrEqual(t, success, 1, "至少应有一次成功")
	assert.LessOrEqual(t, success, 4, "行级保护下成功数不得超过配额 4（无超扣）")
	assert.Equal(t, success, finalUsed, "账目守恒：用量 == 成功扣减数（无丢失/凭空增量）")
}

// ============ 缓存一致性：扣减路径不触发缓存失效（Del），改以精确值对齐 ============

// 缓存成功路径：DECR 后的计数器值即权威剩余量，无需 Delete 重建（避免击穿/重建风暴）。
// 本用例断言该路径只做 Get/Set/Decr、从不调用 cache.Delete；
// DB 降级路径的缓存对齐（Set 精确剩余值）由 DecrError_FallsBackToDB 覆盖。
func TestCheckAndDeductQuota_CachePath_NoDeleteOnSuccess(t *testing.T) {
	// 缓存成功路径经 lifecycle.Go 异步落库，必须多连接（pool）避免单连接协议错乱。
	db := testutil.NewPoolTestDB(t)
	userID := int64(910016)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	cache := servicemocks.NewMockICache(t)
	counterKey := member.MemberQuotaKey(userID, member.QuotaTypeDehaze)
	// 仅 Get/Set/Decr：不 EXPECT Delete，验证扣减成功路径不调用缓存失效。
	cache.EXPECT().Get(mock.Anything, counterKey).Return("", nil).Once()
	cache.EXPECT().Set(mock.Anything, counterKey, int64(10), mock.Anything).Return(nil).Once()
	cache.EXPECT().Decr(mock.Anything, counterKey).Return(int64(9), nil).Once()

	svc := newQuotaService(t, db, cache)
	require.NoError(t, svc.CheckAndDeductQuota(context.Background(), userID, member.QuotaTypeDehaze))
	waitForQuotaUsed(t, db, userID, member.QuotaTypeDehaze, 1)
	assert.Equal(t, 1, memberUsed(t, db, userID, member.QuotaTypeDehaze), "扣减成功且未触发缓存失效")
}

// ============ 行级保护：绕过应用层预校验直接并发扣减 ============

// 直接并发调用 repository 行级条件扣减（跳过 CheckAndDeductQuota 的应用层预校验），
// 权威判定完全依赖 SQL 的 WHERE used < quota：并发成功数不得超过配额，无超扣。
func TestDeductQuotaIfAvailable_Concurrent_RowLevelGuard_NoOversell(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(920004)
	mustCreateMember(t, db, newQuotaMember(userID, 3, 0, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	const workers = 10
	repo := memberrepo.NewMemberRepository(db)

	var wg sync.WaitGroup
	var mu sync.Mutex
	success := 0
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, deducted, err := repo.DeductQuotaIfAvailable(context.Background(), userID, string(member.QuotaTypeDehaze))
			if err != nil {
				t.Errorf("行级条件扣减失败: %v", err)
				return
			}
			if deducted {
				mu.Lock()
				success++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	finalUsed := memberUsed(t, db, userID, member.QuotaTypeDehaze)
	assert.Equal(t, 3, success, "行级保护下成功数恰为配额 3，无超扣")
	assert.Equal(t, 3, finalUsed, "最终用量精确 = 配额，无超扣/少扣")
}

// ============ RefundQuota：used=0 回补不越界 ============

// used 已为 0 时回补：下限保护（used > 0 才更新）使其为 no-op，
// 不报错、不越界，monthly_*_used 恒 ≥0，杜绝凭空制造可扣减额度。
func TestRefundQuota_ZeroUsedRefund_NoOp(t *testing.T) {
	db := testutil.NewTestDB(t)
	resetMemberTables(t, db)
	userID := int64(910017)
	// used 初始已为 0。
	mustCreateMember(t, db, newQuotaMember(userID, 10, 0, 0, 0, 1))

	svc := newQuotaService(t, db, nil)
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	require.NoError(t, svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze))
	assert.Equal(t, 0, memberUsed(t, db, userID, member.QuotaTypeDehaze), "used=0 时回补为 no-op，不得减为负")
}

// ============ 并发回补：账目守恒 ============

// 多 goroutine 并发回补同一用户，断言最终用量 = 初始 - 成功回补数，
// 无丢失更新（UPDATE 行级原子性，counterKey 并发 Incr 由 mock 单连接串行化）。
func TestRefundQuota_Concurrent_AccountingConserved(t *testing.T) {
	db := testutil.NewPoolTestDB(t)
	userID := int64(920003)
	mustCreateMember(t, db, newQuotaMember(userID, 10, 8, 0, 0, 1))
	t.Cleanup(func() {
		require.NoError(t, db.Exec("DELETE FROM sys_member WHERE user_id = ?", userID).Error)
	})

	const workers = 8
	svc := newQuotaService(t, db, nil)

	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = svc.RefundQuota(context.Background(), userID, member.QuotaTypeDehaze)
		}()
	}
	wg.Wait()

	// 初始 used=8，8 次回补各 -1，无下限保护：8 - 8 = 0（恰好不越界）。
	finalUsed := memberUsed(t, db, userID, member.QuotaTypeDehaze)
	assert.Equal(t, 0, finalUsed, "8 次并发回补后用量应精确 = 初始8 - 8 = 0，无丢失更新")
}
