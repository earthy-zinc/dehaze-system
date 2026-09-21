package ai

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"gorm.io/gorm"
)

// BillingRepository AI 计费域数据访问：计费记录 / 余额流水 / 退款申请 / 异常事件 / 余额乐观锁。
// 计费记录、流水、退款、异常均为只追加表（无逻辑删除）；余额落在共享表 sys_user。
type BillingRepository struct {
	db *gorm.DB
}

func NewBillingRepository(db *gorm.DB) *BillingRepository {
	return &BillingRepository{db: db}
}

// ==================== 计费记录 ====================

// PaginateBillingByUser 计费明细分页（create_time/id 倒序）
func (r *BillingRepository) PaginateBillingByUser(
	ctx context.Context, userID int64, page, size int,
	conversationID *int64, billType, modelID string, start, end *time.Time,
) ([]model.SysAiBilling, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).Where("user_id = ?", userID)
	if conversationID != nil {
		db = db.Where("conversation_id = ?", *conversationID)
	}
	if billType != "" {
		db = db.Where("bill_type = ?", billType)
	}
	if modelID != "" {
		db = db.Where("model = ?", modelID)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiBilling
	err := db.Order("create_time DESC, id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// GetBilling 按主键查询计费记录
func (r *BillingRepository) GetBilling(ctx context.Context, id int64) (*model.SysAiBilling, error) {
	var item model.SysAiBilling
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&item).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &item, err
}

// LatestRefundStatusByBillingIDs 批量取计费记录的最新退款状态（id 递增覆盖，0 表示无申请）
func (r *BillingRepository) LatestRefundStatusByBillingIDs(ctx context.Context, ids []int64) (map[int64]int, error) {
	status := make(map[int64]int, len(ids))
	if len(ids) == 0 {
		return status, nil
	}
	var rows []struct {
		BillingID int64 `gorm:"column:billing_id"`
		Status    int   `gorm:"column:status"`
	}
	err := r.db.WithContext(ctx).Model(&model.SysAiRefund{}).
		Select("billing_id, status").
		Where("billing_id IN ?", ids).
		Order("id ASC").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		status[row.BillingID] = row.Status
	}
	return status, nil
}

// ListBillingByRequestID 按 request_id 查询计费记录（过程链详情优先关联口径）
func (r *BillingRepository) ListBillingByRequestID(ctx context.Context, requestID string) ([]model.SysAiBilling, error) {
	var items []model.SysAiBilling
	err := r.db.WithContext(ctx).Where("request_id = ?", requestID).Order("id ASC").Find(&items).Error
	return items, err
}

// ListBillingByMessage 按 message_id 查询计费记录
func (r *BillingRepository) ListBillingByMessage(ctx context.Context, messageID int64) ([]model.SysAiBilling, error) {
	var items []model.SysAiBilling
	err := r.db.WithContext(ctx).Where("message_id = ?", messageID).Order("id ASC").Find(&items).Error
	return items, err
}

// ListBillingByConversation 按会话查询计费记录（会话时间线数据源）
func (r *BillingRepository) ListBillingByConversation(ctx context.Context, conversationID int64) ([]model.SysAiBilling, error) {
	var items []model.SysAiBilling
	err := r.db.WithContext(ctx).Where("conversation_id = ?", conversationID).Order("id ASC").Find(&items).Error
	return items, err
}

// ==================== 统计聚合 ====================

// BillingTypeSum 按 bill_type 汇总
type BillingTypeSum struct {
	BillType     string `gorm:"column:bill_type"`
	Credits      int64  `gorm:"column:credits"`
	InputTokens  int64  `gorm:"column:input_tokens"`
	OutputTokens int64  `gorm:"column:output_tokens"`
	CreditsSaved int64  `gorm:"column:credits_saved"`
}

// SumBillingByBillType 月结账单：按 bill_type 汇总消耗
func (r *BillingRepository) SumBillingByBillType(ctx context.Context, userID int64, start, end time.Time) ([]BillingTypeSum, error) {
	var rows []BillingTypeSum
	err := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).
		Select(`bill_type,
			COALESCE(SUM(credits), 0) AS credits,
			COALESCE(SUM(input_tokens), 0) AS input_tokens,
			COALESCE(SUM(output_tokens), 0) AS output_tokens,
			COALESCE(SUM(credits_saved), 0) AS credits_saved`).
		Where("user_id = ? AND create_time >= ? AND create_time <= ?", userID, start, end).
		Group("bill_type").Scan(&rows).Error
	return rows, err
}

// BillingPeriodSum 按日/月聚合的消耗趋势行
type BillingPeriodSum struct {
	Date              string `gorm:"column:date"`
	Credits           int64  `gorm:"column:credits"`
	InputTokens       int64  `gorm:"column:input_tokens"`
	OutputTokens      int64  `gorm:"column:output_tokens"`
	CreditsSaved      int64  `gorm:"column:credits_saved"`
	CachedInputTokens int64  `gorm:"column:cached_input_tokens"`
}

// SumBillingByPeriod 用户端消耗趋势（day: %Y-%m-%d；month: %Y-%m）
func (r *BillingRepository) SumBillingByPeriod(
	ctx context.Context, userID int64, start, end time.Time, period string, billTypes []string,
) ([]BillingPeriodSum, error) {
	format := "%Y-%m-%d"
	if period == "month" {
		format = "%Y-%m"
	}
	var rows []BillingPeriodSum
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).
		Select(`DATE_FORMAT(create_time, ?) AS date,
			COALESCE(SUM(credits), 0) AS credits,
			COALESCE(SUM(input_tokens), 0) AS input_tokens,
			COALESCE(SUM(output_tokens), 0) AS output_tokens,
			COALESCE(SUM(credits_saved), 0) AS credits_saved,
			COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens`, format).
		Where("user_id = ? AND create_time >= ? AND create_time <= ?", userID, start, end)
	if len(billTypes) > 0 {
		db = db.Where("bill_type IN ?", billTypes)
	}
	err := db.Group("date").Order("date ASC").Scan(&rows).Error
	return rows, err
}

// BillingModelSum 按模型聚合行
type BillingModelSum struct {
	Model            string `gorm:"column:model"`
	Credits          int64  `gorm:"column:credits"`
	InputTokens      int64  `gorm:"column:input_tokens"`
	OutputTokens     int64  `gorm:"column:output_tokens"`
	CreditsSaved     int64  `gorm:"column:credits_saved"`
	DegradationCount int64  `gorm:"column:degradation_count"`
}

// SumBillingByModel 用户端模型消耗分布（降级次数按 actual_model 非空计）
func (r *BillingRepository) SumBillingByModel(
	ctx context.Context, userID int64, start, end time.Time, billTypes []string,
) ([]BillingModelSum, error) {
	var rows []BillingModelSum
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).
		Select(`model,
			COALESCE(SUM(credits), 0) AS credits,
			COALESCE(SUM(input_tokens), 0) AS input_tokens,
			COALESCE(SUM(output_tokens), 0) AS output_tokens,
			COALESCE(SUM(credits_saved), 0) AS credits_saved,
			COALESCE(SUM(CASE WHEN actual_model IS NOT NULL THEN 1 ELSE 0 END), 0) AS degradation_count`).
		Where("user_id = ? AND create_time >= ? AND create_time <= ?", userID, start, end)
	if len(billTypes) > 0 {
		db = db.Where("bill_type IN ?", billTypes)
	}
	err := db.Group("model").Scan(&rows).Error
	return rows, err
}

// BillingDimStat 管理员分维度统计行
type BillingDimStat struct {
	Dimension        string `gorm:"column:dimension"`
	TotalCredits     int64  `gorm:"column:total_credits"`
	TotalInputTokens int64  `gorm:"column:total_input_tokens"`
	TotalOutputToks  int64  `gorm:"column:total_output_tokens"`
	ChatCachedTokens int64  `gorm:"column:chat_cached_tokens"`
	CreditsSaved     int64  `gorm:"column:credits_saved"`
	DegradationCount int64  `gorm:"column:degradation_count"`
}

// StatsBillingByDimension 管理员分维度统计（user/model/billType/day）
//
// token 相关列仅基于 chat 类记录：asr/tts 的 input_tokens 存秒数/字符数，混入会使 token 统计失真。
func (r *BillingRepository) StatsBillingByDimension(
	ctx context.Context, groupBy string, userID *int64, modelID, billType string, start, end *time.Time,
) ([]BillingDimStat, error) {
	dimExpr, err := billingDimExpr(groupBy)
	if err != nil {
		return nil, err
	}
	chatTypes := []string{"chat", "chat_subagent"}
	var rows []BillingDimStat
	// dimExpr 来自枚举分支且不含占位符，拼接无注入面；chat 类过滤走 ? 绑定
	selectClause := fmt.Sprintf(`%s AS dimension,`, dimExpr)
	db := r.db.WithContext(ctx).Model(&model.SysAiBilling{}).
		Select(selectClause+`
			COALESCE(SUM(credits), 0) AS total_credits,
			COALESCE(SUM(CASE WHEN bill_type IN ? THEN input_tokens ELSE 0 END), 0) AS total_input_tokens,
			COALESCE(SUM(CASE WHEN bill_type IN ? THEN output_tokens ELSE 0 END), 0) AS total_output_tokens,
			COALESCE(SUM(CASE WHEN bill_type IN ? THEN cached_input_tokens ELSE 0 END), 0) AS chat_cached_tokens,
			COALESCE(SUM(credits_saved), 0) AS credits_saved,
			COALESCE(SUM(CASE WHEN actual_model IS NOT NULL THEN 1 ELSE 0 END), 0) AS degradation_count`,
			chatTypes, chatTypes, chatTypes)
	if userID != nil {
		db = db.Where("user_id = ?", *userID)
	}
	if modelID != "" {
		db = db.Where("model = ?", modelID)
	}
	if billType != "" {
		db = db.Where("bill_type = ?", billType)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	err = db.Group("dimension").Order("dimension ASC").Scan(&rows).Error
	return rows, err
}

// billingDimExpr 统计维度表达式（与 python stats_by_dimension 一致，枚举外维度报错）
func billingDimExpr(groupBy string) (string, error) {
	switch groupBy {
	case "user":
		return "user_id", nil
	case "model":
		return "model", nil
	case "billType":
		return "bill_type", nil
	case "day":
		return "DATE_FORMAT(create_time, '%Y-%m-%d')", nil
	default:
		// python 侧对非法维度抛 ValueError（未映射业务码 → B0001），此处保持一致
		return "", fmt.Errorf("不支持的统计维度: %s", groupBy)
	}
}

// ==================== 余额流水 ====================

// PaginateCreditLogs 余额流水分页（create_time/id 倒序）
func (r *BillingRepository) PaginateCreditLogs(
	ctx context.Context, userID int64, page, size int, source string, start, end *time.Time,
) ([]model.SysAiCreditLog, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiCreditLog{}).Where("user_id = ?", userID)
	if source != "" {
		db = db.Where("source = ?", source)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiCreditLog
	err := db.Order("create_time DESC, id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// SumCreditLogBySource 账单口径：按 source 汇总金额变动
func (r *BillingRepository) SumCreditLogBySource(ctx context.Context, userID int64, start, end *time.Time) (map[string]int64, error) {
	var rows []struct {
		Source string `gorm:"column:source"`
		Amount int64  `gorm:"column:amount"`
	}
	db := r.db.WithContext(ctx).Model(&model.SysAiCreditLog{}).
		Select("source, COALESCE(SUM(amount), 0) AS amount").
		Where("user_id = ?", userID)
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	if err := db.Group("source").Scan(&rows).Error; err != nil {
		return nil, err
	}
	result := make(map[string]int64, len(rows))
	for _, row := range rows {
		result[row.Source] = row.Amount
	}
	return result, nil
}

// BalanceAtOrBefore 指定时刻前最近一笔流水的变动后余额（账单期初/期末余额）
func (r *BillingRepository) BalanceAtOrBefore(ctx context.Context, userID int64, at time.Time) (int64, error) {
	var balance int64
	err := r.db.WithContext(ctx).Model(&model.SysAiCreditLog{}).
		Select("balance_after").
		Where("user_id = ? AND create_time <= ?", userID, at).
		Order("create_time DESC, id DESC").Limit(1).
		Scan(&balance).Error
	return balance, err
}

// CreateCreditLog 追加余额流水
func (r *BillingRepository) CreateCreditLog(ctx context.Context, log *model.SysAiCreditLog) error {
	return r.db.WithContext(ctx).Create(log).Error
}

// ==================== 退款申请 ====================

// PaginateRefunds 退款申请分页（管理端审核，create_time/id 倒序）
func (r *BillingRepository) PaginateRefunds(
	ctx context.Context, page, size int, status *int, userID *int64, start, end *time.Time,
) ([]model.SysAiRefund, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiRefund{})
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	if userID != nil {
		db = db.Where("user_id = ?", *userID)
	}
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiRefund
	err := db.Order("create_time DESC, id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// GetRefund 按主键查询退款申请
func (r *BillingRepository) GetRefund(ctx context.Context, id int64) (*model.SysAiRefund, error) {
	var item model.SysAiRefund
	err := r.db.WithContext(ctx).Where("id = ?", id).First(&item).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &item, err
}

// GetPendingRefundByBilling 查询计费记录下未完结的退款申请（防重复申请）
func (r *BillingRepository) GetPendingRefundByBilling(ctx context.Context, billingID int64) (*model.SysAiRefund, error) {
	var item model.SysAiRefund
	err := r.db.WithContext(ctx).Where("billing_id = ? AND status = 1", billingID).Limit(1).Find(&item).Error
	if err != nil {
		return nil, err
	}
	if item.ID == 0 {
		return nil, nil
	}
	return &item, nil
}

// HasApprovedRefund 是否已存在审核通过的退款（同一计费记录仅允许一次补偿）
func (r *BillingRepository) HasApprovedRefund(ctx context.Context, billingID, excludeID int64) (bool, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiRefund{}).
		Where("billing_id = ? AND status = 2", billingID)
	if excludeID > 0 {
		db = db.Where("id != ?", excludeID)
	}
	var count int64
	err := db.Count(&count).Error
	return count > 0, err
}

// CreateRefund 新增退款申请
func (r *BillingRepository) CreateRefund(ctx context.Context, refund *model.SysAiRefund) error {
	return r.db.WithContext(ctx).Create(refund).Error
}

// UpdateRefund 更新退款申请（状态流转与审核信息）
func (r *BillingRepository) UpdateRefund(ctx context.Context, id int64, fields map[string]any) error {
	return r.db.WithContext(ctx).Model(&model.SysAiRefund{}).Where("id = ?", id).Updates(fields).Error
}

// ==================== 异常事件 ====================

// PaginateAnomalies 异常事件分页（trigger_at/id 倒序）
func (r *BillingRepository) PaginateAnomalies(
	ctx context.Context, page, size int, userID *int64, anomalyType string, status *int, start, end *time.Time,
) ([]model.SysAiBillingAnomaly, int64, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiBillingAnomaly{})
	if userID != nil {
		db = db.Where("user_id = ?", *userID)
	}
	if anomalyType != "" {
		db = db.Where("anomaly_type = ?", anomalyType)
	}
	if status != nil {
		db = db.Where("status = ?", *status)
	}
	if start != nil {
		db = db.Where("trigger_at >= ?", *start)
	}
	if end != nil {
		// 上界排他（与 python list_page 的 trigger_at < date_end 一致）
		db = db.Where("trigger_at < ?", *end)
	}
	var total int64
	if err := db.Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiBillingAnomaly
	err := db.Order("trigger_at DESC, id DESC").Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// ==================== 用户余额（sys_user 乐观锁 CAS） ====================

// UserBalance 余额与乐观锁版本
type UserBalance struct {
	Balance float64 `gorm:"column:credits_balance"`
	Version int     `gorm:"column:credits_version"`
}

// GetUserBalance 读取余额与版本号，用户不存在或已删返回 nil
func (r *BillingRepository) GetUserBalance(ctx context.Context, userID int64) (*UserBalance, error) {
	var row UserBalance
	err := r.db.WithContext(ctx).Table("sys_user").
		Select("credits_balance, credits_version").
		Where("id = ? AND deleted = 0", userID).Limit(1).Scan(&row).Error
	if err != nil {
		return nil, err
	}
	if row.Balance == 0 && row.Version == 0 {
		// 需区分"余额为 0 的真实账户"与"用户不存在"，用存在性单独判定
		var count int64
		if err := r.db.WithContext(ctx).Table("sys_user").
			Where("id = ? AND deleted = 0", userID).Count(&count).Error; err != nil {
			return nil, err
		}
		if count == 0 {
			return nil, nil
		}
	}
	return &row, nil
}

// ExistsUser 用户是否存在（未软删）
func (r *BillingRepository) ExistsUser(ctx context.Context, userID int64) (bool, error) {
	var count int64
	err := r.db.WithContext(ctx).Table("sys_user").
		Where("id = ? AND deleted = 0", userID).Count(&count).Error
	return count > 0, err
}

// AddBalanceCAS 乐观锁增加余额（version 不匹配返回 false，由调用方重试）
func (r *BillingRepository) AddBalanceCAS(ctx context.Context, userID int64, amount int64, version int) (bool, error) {
	result := r.db.WithContext(ctx).Exec(
		"UPDATE sys_user SET credits_balance = credits_balance + ?, credits_version = credits_version + 1 WHERE id = ? AND credits_version = ?",
		amount, userID, version,
	)
	if result.Error != nil {
		return false, result.Error
	}
	return result.RowsAffected == 1, nil
}

// QuotaLimits 用户日/月积分限额：经会员等级取启用（status=1）权益；
// 无会员、权益缺失或已停用时返回 nil（fail-closed，配额校验一律拒绝）
func (r *BillingRepository) QuotaLimits(ctx context.Context, userID int64) (*[2]int64, error) {
	var rows []struct {
		Daily   int64 `gorm:"column:ai_credits_daily"`
		Monthly int64 `gorm:"column:ai_credits_monthly"`
	}
	err := r.db.WithContext(ctx).Table("sys_member AS m").
		Select("b.ai_credits_daily, b.ai_credits_monthly").
		Joins("JOIN sys_member_benefit AS b ON b.level_code = m.level_code AND b.deleted = 0 AND b.status = 1").
		Where("m.user_id = ? AND m.deleted = 0", userID).
		Limit(1).Find(&rows).Error
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, nil
	}
	return &[2]int64{rows[0].Daily, rows[0].Monthly}, nil
}
