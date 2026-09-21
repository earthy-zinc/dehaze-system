package ai

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	airepo "github.com/earthyzinc/dehaze-go/internal/repository/ai"
	"github.com/earthyzinc/dehaze-go/internal/testutil"
	"github.com/earthyzinc/dehaze-go/pkg/common"
	goredis "github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
)

// ==================== 余额与欠费 ====================

func TestBillingBalanceView(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, cache := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 100)

	balance, err := svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, "100.00", balance.CreditsBalance, "缓存未命中：查 MySQL 后回填并整数化")
	require.False(t, balance.ArrearsStatus)
	require.EqualValues(t, 0, balance.DailyUsed)

	// 回填值必须是整数字符串，否则后续 DECRBY 会因 \"100.00\" 报错
	cached, err := cache.Get(fmt.Sprintf("ai:balance:%d", userID))
	require.NoError(t, err)
	require.Equal(t, "100", cached)

	balance, err = svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, "100", balance.CreditsBalance, "缓存命中直接返回原始值（与 python Decimal(str) 序列化一致）")

	// 缓存坏值：删除后按 MySQL 值重建
	require.NoError(t, cache.Set(fmt.Sprintf("ai:balance:%d", userID), "not-a-number"))
	balance, err = svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, "100.00", balance.CreditsBalance)

	require.NoError(t, cache.Set(fmt.Sprintf("ai:arrears:%d", userID), "1"))
	balance, err = svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.True(t, balance.ArrearsStatus)

	dailyKey := fmt.Sprintf("ai:quota:daily:%d:%s", userID, time.Now().In(shanghaiTZ).Format("2006-01-02"))
	monthKey := fmt.Sprintf("ai:quota:monthly:%d:%s", userID, time.Now().In(shanghaiTZ).Format("2006-01"))
	require.NoError(t, cache.Set(dailyKey, "30"))
	require.NoError(t, cache.Set(monthKey, "77"))
	balance, err = svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.EqualValues(t, 30, balance.DailyUsed)
	require.EqualValues(t, 77, balance.MonthlyUsed)
}

func TestBillingAdjust(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, cache := newBillingServiceForTest(t, db)
	ctx := context.Background()
	operatorID := seedBillingUser(t, db, 0)

	result, err := svc.Adjust(ctx, operatorID, &bo.CreditAdjustForm{
		UserID: operatorID, Amount: 50, Reason: "运营补偿",
	})
	require.NoError(t, err)
	require.Equal(t, "50", result.CreditsBalance)

	var creditsBalance float64
	require.NoError(t, db.Table("sys_user").Select("credits_balance").
		Where("id = ?", operatorID).Scan(&creditsBalance).Error)
	require.EqualValues(t, 50, creditsBalance, "余额需落库（Redis 仅为实时权威视图）")

	var logs []model.SysAiCreditLog
	require.NoError(t, db.Where("user_id = ? AND source = ?", operatorID, "admin_adjust").
		Find(&logs).Error)
	require.Len(t, logs, 1)
	require.EqualValues(t, 50, logs[0].Amount)
	require.EqualValues(t, 50, logs[0].BalanceAfter)
	require.Equal(t, "运营补偿", *logs[0].Reason)
	require.Equal(t, operatorID, *logs[0].OperatorID)

	// 欠费标记在余额增加时清除
	require.NoError(t, cache.Set(fmt.Sprintf("ai:arrears:%d", operatorID), "1"))
	_, err = svc.Adjust(ctx, operatorID, &bo.CreditAdjustForm{UserID: operatorID, Amount: 1, Reason: "补零"})
	require.NoError(t, err)
	require.False(t, cache.Exists(fmt.Sprintf("ai:arrears:%d", operatorID)))

	_, err = svc.Adjust(ctx, operatorID, &bo.CreditAdjustForm{UserID: operatorID, Amount: 0, Reason: "零值"})
	requireBizCode(t, err, common.PARAM_ERROR)

	_, err = svc.Adjust(ctx, operatorID, &bo.CreditAdjustForm{
		UserID: operatorID + 999999, Amount: 10, Reason: "不存在",
	})
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
}

// ==================== 退款申请与审核 ====================

func TestBillingRefundFlow(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _ := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 0)
	record := seedBillingRecord(t, db, userID, 20, "chat")

	// 记录归属校验：他人记录不可申请
	_, err := svc.ApplyRefund(ctx, userID+999999, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 5, Reason: "误扣",
	})
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	// 金额校验
	_, err = svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 0, Reason: "误扣",
	})
	requireBizCode(t, err, common.PARAM_ERROR)

	_, err = svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 21, Reason: "误扣",
	})
	requireBizCode(t, err, common.PARAM_ERROR)

	refund, err := svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 20, Reason: "误扣",
	})
	require.NoError(t, err)
	require.Equal(t, 1, refund.Status, "新建退款申请为待审核")

	_, err = svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 20, Reason: "误扣",
	})
	requireBizCode(t, err, common.AI_REFUND_ALREADY_EXISTS)

	// 计费明细透出最新退款申请状态（分页缺省由 handler 解析，service 只按透传值分页）
	page, err := svc.Records(ctx, userID, &bo.BillingRecordQuery{PageNum: 1, PageSize: 20})
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total)
	require.Equal(t, 1, page.List[0].RefundStatus)

	// 审核通过：余额回补 + 流水 source=refund
	audited, err := svc.AuditRefund(ctx, refund.ID, true, strPtr("核实无误"), userID)
	require.NoError(t, err)
	require.Equal(t, 2, audited.Status)
	require.Equal(t, userID, *audited.AuditorID)

	var refundLogs []model.SysAiCreditLog
	require.NoError(t, db.Where("user_id = ? AND source = ?", userID, "refund").Find(&refundLogs).Error)
	require.Len(t, refundLogs, 1)
	require.EqualValues(t, 20, refundLogs[0].Amount)
	require.EqualValues(t, record.ID, *refundLogs[0].RelatedID)

	balance, err := svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, "20", balance.CreditsBalance, "审核通过后余额回补")

	// 已审核不可重复处理
	_, err = svc.AuditRefund(ctx, refund.ID, true, nil, userID)
	requireBizCode(t, err, common.REFUND_AUDIT_FAILED)

	_, err = svc.AuditRefund(ctx, refund.ID+999999, true, nil, userID)
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	refunds, err := svc.ListRefunds(ctx, &bo.RefundQuery{PageNum: 1, PageSize: 20})
	require.NoError(t, err)
	require.EqualValues(t, 1, refunds.Total)
	require.Equal(t, 2, refunds.List[0].Status)

	invalidStatus := 4
	_, err = svc.ListRefunds(ctx, &bo.RefundQuery{Status: &invalidStatus})
	requireBizCode(t, err, common.PARAM_ERROR)
}

func TestBillingRefundRejectKeepsBalance(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _ := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 0)
	record := seedBillingRecord(t, db, userID, 9, "chat")

	refund, err := svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 9, Reason: "误扣",
	})
	require.NoError(t, err)

	rejected, err := svc.AuditRefund(ctx, refund.ID, false, strPtr("计费无误"), userID)
	require.NoError(t, err)
	require.Equal(t, 3, rejected.Status)

	balance, err := svc.Balance(ctx, userID)
	require.NoError(t, err)
	require.Equal(t, "0.00", balance.CreditsBalance, "驳回不回补余额")

	// 驳回后允许再次申请
	_, err = svc.ApplyRefund(ctx, userID, &bo.BillingRefundApplyForm{
		BillingID: record.ID, Amount: 9, Reason: "二次申诉",
	})
	require.NoError(t, err)
}

// ==================== 汇总与账单 ====================

func TestBillingSummaryAndBill(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _ := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 0)
	seedBillingRecord(t, db, userID, 30, "chat")
	seedBillingRecord(t, db, userID, 12, "chat_subagent")
	seedBillingRecord(t, db, userID, 99, "tts")

	summary, err := svc.Summary(ctx, userID, "day")
	require.NoError(t, err)
	require.EqualValues(t, 42, summary.TotalCredits, "token 口径仅统计 chat 类，tts 不计入")
	require.EqualValues(t, 200, summary.InputTokens)
	require.EqualValues(t, 80, summary.OutputTokens)
	require.EqualValues(t, 20, summary.Savings.CachedInputTokens)
	require.EqualValues(t, 4, summary.Savings.CreditsSaved)
	require.Len(t, summary.ModelDistribution, 1)

	_, err = svc.Summary(ctx, userID, "hour")
	requireBizCode(t, err, common.PARAM_ERROR)

	month := time.Now().Format("2006-01")
	bill, err := svc.Bill(ctx, userID, month)
	require.NoError(t, err)
	require.Equal(t, month, bill.Month)
	require.EqualValues(t, 141, bill.TotalConsume)
	require.EqualValues(t, 99, bill.ItemSummary["tts"])
	require.Equal(t, "0", bill.BalanceEnd)

	_, err = svc.Bill(ctx, userID, "2020-01")
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	// 非零填充月份与 "2020-01" 同月（python strptime("%Y-%m") 口径，SDK 集成用例同样钉这条），
	// 故走的是"空账期"而非"格式错误"
	_, err = svc.Bill(ctx, userID, "2020-1")
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	_, err = svc.Bill(ctx, userID, "2020-13")
	requireBizCode(t, err, common.PARAM_ERROR)
}

func TestBillingStatsByDimension(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, _ := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 0)
	seedBillingRecord(t, db, userID, 30, "chat")
	seedBillingRecord(t, db, userID, 12, "chat_subagent")

	stats, err := svc.Stats(ctx, &bo.BillingStatQuery{GroupBy: "billType"})
	require.NoError(t, err)
	require.NotEmpty(t, stats)
	var chatRow bool
	for _, row := range stats {
		if row.Dimension == "chat" {
			chatRow = true
			require.EqualValues(t, 100, row.TotalInputTokens)
			require.EqualValues(t, 0.1, row.CacheHitRate, "缓存命中率 = chat 缓存 token / chat 输入 token")
		}
	}
	require.True(t, chatRow)

	stats, err = svc.Stats(ctx, &bo.BillingStatQuery{GroupBy: "day"})
	require.NoError(t, err)
	require.NotEmpty(t, stats)

	_, err = svc.Stats(ctx, &bo.BillingStatQuery{GroupBy: "hour"})
	requireBizCode(t, err, common.PARAM_ERROR)
}

// ==================== 成本单价与成本统计 ====================

func TestCostServiceCRUD(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewCostService(airepo.NewCostRepository(db))
	ctx := context.Background()
	operatorID := seedBillingUser(t, db, 0)
	modelID := fmt.Sprintf("m-cost-%d", time.Now().UnixNano())

	created, err := svc.CreateCost(ctx, operatorID, &bo.ModelCostCreateForm{
		ModelID: modelID, ProviderID: 1,
		Details: []bo.ModelCostDetailForm{{
			TokenType: "input", TimeSlot: "default", UnitPrice: 0.0012,
		}},
	})
	require.NoError(t, err)
	require.Equal(t, 1, created.PriceVersion)
	require.Equal(t, "CNY", created.Currency)
	require.Len(t, created.Details, 1)
	require.Equal(t, "0.0012", created.Details[0].UnitPrice, "unit_price 为 decimal(12,4) 字符串")

	second, err := svc.CreateCost(ctx, operatorID, &bo.ModelCostCreateForm{
		ModelID: modelID, ProviderID: 1,
	})
	require.NoError(t, err)
	require.Equal(t, 2, second.PriceVersion, "同模型同供应商追加价格版本")

	page, err := svc.ListCosts(ctx, &bo.ModelCostQuery{PageNum: 1, PageSize: 20, Keyword: modelID})
	require.NoError(t, err)
	require.EqualValues(t, 2, page.Total)
	require.ElementsMatch(t, []int{1, 2},
		[]int{page.List[0].PriceVersion, page.List[1].PriceVersion}, "两个价格版本都可见")

	status := 0
	updated, err := svc.UpdateCost(ctx, created.ID, operatorID, &bo.ModelCostUpdateForm{Status: &status})
	require.NoError(t, err)
	require.Equal(t, 0, updated.Status)
	require.Len(t, updated.Details, 1, "更新主表不影响档位明细")

	_, err = svc.UpdateCost(ctx, created.ID+999999, operatorID, &bo.ModelCostUpdateForm{})
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	require.NoError(t, svc.DeleteCost(ctx, second.ID, operatorID))
	page, err = svc.ListCosts(ctx, &bo.ModelCostQuery{PageNum: 1, PageSize: 20, Keyword: modelID})
	require.NoError(t, err)
	require.EqualValues(t, 1, page.Total)

	requireBizCode(t, svc.DeleteCost(ctx, second.ID, operatorID), common.RESOURCE_NOT_FOUND)

	stats, err := svc.CostStats(ctx, &bo.CostStatQuery{GroupBy: "overall"})
	require.NoError(t, err)
	require.Len(t, stats, 2, "整体口径返回 overall 与 ai 两行毛利")
	require.Equal(t, "overall", *stats[0].Metric)
	require.NotNil(t, stats[0].Revenue)

	stats, err = svc.CostStats(ctx, &bo.CostStatQuery{GroupBy: "model"})
	require.NoError(t, err)
	require.NotNil(t, stats)

	_, err = svc.CostStats(ctx, &bo.CostStatQuery{GroupBy: "hour"})
	requireBizCode(t, err, common.PARAM_ERROR)
}

// TestBillCacheCrossEndInterop 账单缓存三端互认：key `ai:bill:{uid}:{month}` 由三端共享，
// python 写 snake_case + Decimal→字符串、java 写 snake_case + BigDecimal→数字，
// go 必须两种都能读、且自己写出的是 python/java 都能读的 snake_case 字符串形态。
//
// 用"无任何库内记录的月份"作夹具：读缓存成功则返回夹具值，读失败会重新生成 → A0401（空账期），
// 据此可判定缓存是否真被读到（自写自读无法暴露格式断裂）。
func TestBillCacheCrossEndInterop(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc, cache := newBillingServiceForTest(t, db)
	ctx := context.Background()
	userID := seedBillingUser(t, db, 0)

	// python 形态夹具（app/service/billing/bill_service.py model_dump(mode="json")）
	pythonPayload := `{"user_id":` + strconv.FormatInt(userID, 10) + `,"month":"2020-01",` +
		`"total_consume":120,"total_recharge":50,"total_refund":7,` +
		`"balance_start":"30.00","balance_end":"73.00",` +
		`"item_summary":{"chat":100,"tts":20}}`
	require.NoError(t, cache.Set(fmt.Sprintf("ai:bill:%d:2020-01", userID), pythonPayload))

	bill, err := svc.Bill(ctx, userID, "2020-01")
	require.NoError(t, err, "python 写的 snake_case 缓存必须能直接读出")
	require.EqualValues(t, 120, bill.TotalConsume)
	require.EqualValues(t, 50, bill.TotalRecharge)
	require.EqualValues(t, 7, bill.TotalRefund)
	require.Equal(t, "30.00", bill.BalanceStart)
	require.Equal(t, "73.00", bill.BalanceEnd)
	require.EqualValues(t, 100, bill.ItemSummary["chat"])
	require.EqualValues(t, 20, bill.ItemSummary["tts"])

	// java 形态夹具（AiBillVO + AiJsonUtils SNAKE_MAPPER：BigDecimal 序列化为数字）
	require.NoError(t, cache.Set(fmt.Sprintf("ai:bill:%d:2020-02", userID),
		`{"user_id":`+strconv.FormatInt(userID, 10)+`,"month":"2020-02",`+
			`"total_consume":9,"total_recharge":0,"total_refund":0,`+
			`"balance_start":0,"balance_end":9.5,"item_summary":{"chat":9}}`))

	bill, err = svc.Bill(ctx, userID, "2020-02")
	require.NoError(t, err, "java 写的数字形态金额必须能直接读出")
	require.EqualValues(t, 9, bill.TotalConsume)
	require.Equal(t, "0", bill.BalanceStart)
	require.Equal(t, "9.5", bill.BalanceEnd)

	// go 写出形态：snake_case + 字符串金额（python 同形、Jackson 可转 BigDecimal）
	seedBillingRecord(t, db, userID, 33, "chat")
	current := time.Now().Format("2006-01")
	_, err = svc.Bill(ctx, userID, current)
	require.NoError(t, err)
	written, err := cache.Get(fmt.Sprintf("ai:bill:%d:%s", userID, current))
	require.NoError(t, err)
	require.Contains(t, written, `"total_consume":33`)
	require.Contains(t, written, `"item_summary":{"chat":33}`)
	require.Contains(t, written, `"balance_start":"`)
	require.NotContains(t, written, `"totalConsume"`, "缓存严禁下发 camelCase 键")
	require.NotContains(t, written, `"itemSummary"`)

	// 三端都写空的月份：空账期一律 A0401（不因读到全 0 缓存而误返回成功）
	require.NoError(t, cache.Set(fmt.Sprintf("ai:bill:%d:2020-03", userID),
		`{"user_id":`+strconv.FormatInt(userID, 10)+`,"month":"2020-03",`+
			`"total_consume":0,"total_recharge":0,"total_refund":0,`+
			`"balance_start":"0","balance_end":"0","item_summary":{}}`))
	_, err = svc.Bill(ctx, userID, "2020-03")
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)

	// 坏值（非 JSON）不影响可用性：重新生成
	require.NoError(t, cache.Set(fmt.Sprintf("ai:bill:%d:2020-04", userID), "not-json"))
	_, err = svc.Bill(ctx, userID, "2020-04")
	requireBizCode(t, err, common.RESOURCE_NOT_FOUND)
}

// TestCostServiceRejectsInvalidForm 成本表单取值校验：python 由 pydantic 拦截的约束
// （model_id 1~64 / currency ≤8 / unit_price ≥0 / min_tokens ≥0）在 go 侧必须于 service 层等价拦截，
// 否则负单价/超长标识会直接落库（DB 报错 → 500，或脏数据进入成本核算）。
func TestCostServiceRejectsInvalidForm(t *testing.T) {
	db := testutil.NewTestDB(t)
	svc := NewCostService(airepo.NewCostRepository(db))
	ctx := context.Background()
	operatorID := seedBillingUser(t, db, 0)

	validDetail := bo.ModelCostDetailForm{TokenType: "input", TimeSlot: "peak", UnitPrice: 0.5, MinTokens: 0}
	cases := []struct {
		name string
		form *bo.ModelCostCreateForm
	}{
		{"空模型标识", &bo.ModelCostCreateForm{ModelID: "", ProviderID: 1}},
		{"模型标识超长", &bo.ModelCostCreateForm{ModelID: strings.Repeat("m", 65), ProviderID: 1}},
		{"币种超长", &bo.ModelCostCreateForm{ModelID: "m-valid", ProviderID: 1, Currency: "CNYCNYCNY9"}},
		{"负单价", &bo.ModelCostCreateForm{ModelID: "m-valid", ProviderID: 1,
			Details: []bo.ModelCostDetailForm{{TokenType: "input", TimeSlot: "peak", UnitPrice: -0.01}}}},
		{"负分段下界", &bo.ModelCostCreateForm{ModelID: "m-valid", ProviderID: 1,
			Details: []bo.ModelCostDetailForm{{TokenType: "input", TimeSlot: "peak", UnitPrice: 0.5, MinTokens: -1}}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := svc.CreateCost(ctx, operatorID, c.form)
			requireBizCode(t, err, common.PARAM_ERROR)
		})
	}

	// 合法表单照常通过（校验不得误伤）
	created, err := svc.CreateCost(ctx, operatorID, &bo.ModelCostCreateForm{
		ModelID: "m-cost-valid-" + strconv.FormatInt(time.Now().UnixNano(), 10), ProviderID: 1,
		Details: []bo.ModelCostDetailForm{validDetail},
	})
	require.NoError(t, err)
	require.Len(t, created.Details, 1)

	// 更新表单的币种同样受限
	invalidCurrency := "CNYCNYCNY9"
	_, err = svc.UpdateCost(ctx, created.ID, operatorID, &bo.ModelCostUpdateForm{Currency: &invalidCurrency})
	requireBizCode(t, err, common.PARAM_ERROR)
}

func TestCostServiceImportReconcile(t *testing.T) {
	svc := NewCostService(nil)
	require.Equal(t, 2, svc.ImportReconcile(&bo.ReconcileImportForm{
		Content: "model,credits\nm1,10\n\n",
	}))
}

// ==================== 辅助 ====================

func newBillingServiceForTest(t *testing.T, db *gorm.DB) (*BillingService, *miniredis.Miniredis) {
	t.Helper()
	cache := miniredis.RunT(t)
	client := goredis.NewClient(&goredis.Options{Addr: cache.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	return NewBillingService(db, airepo.NewBillingRepository(db), client), cache
}

// seedBillingUser 复用种子管理员作为被测账户并重置余额（写入随测试事务回滚）。
func seedBillingUser(t *testing.T, db *gorm.DB, creditsBalance int64) int64 {
	t.Helper()
	var userID int64
	require.NoError(t, db.Table("sys_user").Select("id").
		Where("username = ?", "admin").Limit(1).Scan(&userID).Error)
	require.NotZero(t, userID, "测试库缺少 admin 种子用户")
	require.NoError(t, db.Exec(
		"UPDATE sys_user SET credits_balance = ?, credits_version = 0 WHERE id = ?",
		creditsBalance, userID).Error)
	return userID
}

func seedBillingRecord(t *testing.T, db *gorm.DB, userID int64, credits int, billType string) model.SysAiBilling {
	t.Helper()
	record := model.SysAiBilling{
		UserID: userID, Model: "m-billing-test", BillType: billType,
		InputTokens: 100, CachedInputTokens: 10, OutputTokens: 40,
		Credits: credits, CreditsSaved: 2, QuotaConsumed: credits,
		CreateTime: time.Now(),
	}
	require.NoError(t, db.Create(&record).Error)
	return record
}

// TestMonthBoundsAcceptsNonPaddedMonth 账单月份解析对齐 python strptime("%Y-%m")：非零填充写法
// （"2026-1"）与 "2026-01" 解析为同一月，否则同一请求在 go 报 A0400、在 python 走成空账期 A0401。
func TestMonthBoundsAcceptsNonPaddedMonth(t *testing.T) {
	start, end, err := monthBounds("2026-1")
	require.NoError(t, err)
	require.Equal(t, "2026-01", start.Format("2006-01"))
	require.Equal(t, "2026-01-01", start.Format("2006-01-02"))
	require.Equal(t, 31, end.Day(), "2026-01 月末为 31 日")

	start, end, err = monthBounds("2026-01")
	require.NoError(t, err)
	require.Equal(t, "2026-01-01", start.Format("2006-01-02"))
	require.Equal(t, 31, end.Day())

	for _, invalid := range []string{"2099-13", "2026-", "2026", "invalid-month", "2026-1-5"} {
		_, _, err = monthBounds(invalid)
		requireBizCode(t, err, common.PARAM_ERROR)
	}
}
