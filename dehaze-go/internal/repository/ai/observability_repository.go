package ai

import (
	"context"
	"errors"
	"time"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/model/bo"
	"gorm.io/gorm"
)

// ObservableTraceTypes 配额拒绝类 error_type（采集链路写入的中断/计费拒绝收尾类型）
var quotaRejectErrorTypes = []string{"quota", "quota_exceeded", "precharge_blocked", "arrears", "balance_exceeded"}

// highRiskStepThreshold 高风险调用：推理步数超阈值（防循环观测）
const highRiskStepThreshold = 40

// ObservabilityRepository AI 可观测性查询（sys_ai_trace / sys_ai_llm_call，均为只追加日志表）
type ObservabilityRepository struct {
	db *gorm.DB
}

func NewObservabilityRepository(db *gorm.DB) *ObservabilityRepository {
	return &ObservabilityRepository{db: db}
}

// ==================== 过程链检索 ====================

// TraceFilter 过程链检索条件（时间已由服务层解析）
type TraceFilter struct {
	ConversationID *int64
	UserID         *int64
	Status         *int
	AgentCode      string
	Model          string
	ErrorType      string
	Keyword        string
	Capability     string
	Start          *time.Time
	End            *time.Time
}

// FilterFromQuery 由查询对象构造检索条件
func FilterFromQuery(q *bo.TracePageQuery, start, end *time.Time) TraceFilter {
	return TraceFilter{
		ConversationID: q.ConversationID,
		UserID:         q.UserID,
		Status:         q.Status,
		AgentCode:      q.AgentCode,
		Model:          q.Model,
		ErrorType:      q.ErrorType,
		Keyword:        q.Keyword,
		Capability:     q.Capability,
		Start:          start,
		End:            end,
	}
}

// filteredTraces 应用检索条件（用户归属与标题关键词共用一次会话表关联，避免重复 join 放大）
func (r *ObservabilityRepository) filteredTraces(ctx context.Context, f TraceFilter) *gorm.DB {
	db := r.db.WithContext(ctx).Model(&model.SysAiTrace{})
	if f.ConversationID != nil {
		db = db.Where("sys_ai_trace.conversation_id = ?", *f.ConversationID)
	}
	if f.UserID != nil || f.Keyword != "" {
		db = db.Joins("JOIN sys_ai_conversation AS c ON c.id = sys_ai_trace.conversation_id")
	}
	if f.UserID != nil {
		db = db.Where("c.user_id = ? AND c.deleted = 0", *f.UserID)
	}
	if f.Status != nil {
		db = db.Where("sys_ai_trace.status = ?", *f.Status)
	}
	if f.AgentCode != "" {
		db = db.Where("sys_ai_trace.agent_code = ?", f.AgentCode)
	}
	if f.Model != "" {
		db = db.Where("sys_ai_trace.model = ?", f.Model)
	}
	if f.ErrorType != "" {
		db = db.Where("sys_ai_trace.error_type = ?", f.ErrorType)
	}
	if f.Keyword != "" {
		pattern := "%" + escapeLike(f.Keyword) + "%"
		db = db.Where("(sys_ai_trace.trace_id LIKE ? ESCAPE '\\\\' OR c.title LIKE ? ESCAPE '\\\\')", pattern, pattern)
	}
	if f.Capability != "" {
		// 能力维度：匹配 context_snapshot.items[].type 构成项
		db = db.Where("JSON_SEARCH(sys_ai_trace.context_snapshot, 'one', ?, NULL, '$.items[*].type') IS NOT NULL", f.Capability)
	}
	if f.Start != nil {
		db = db.Where("sys_ai_trace.create_time >= ?", *f.Start)
	}
	if f.End != nil {
		db = db.Where("sys_ai_trace.create_time <= ?", *f.End)
	}
	return db
}

// PaginateTraces 过程链分页检索（create_time/id 倒序）
func (r *ObservabilityRepository) PaginateTraces(ctx context.Context, f TraceFilter, page, size int) ([]model.SysAiTrace, int64, error) {
	var total int64
	if err := r.filteredTraces(ctx, f).Count(&total).Error; err != nil {
		return nil, 0, err
	}
	var items []model.SysAiTrace
	err := r.filteredTraces(ctx, f).Select("sys_ai_trace.*").
		Order("sys_ai_trace.create_time DESC, sys_ai_trace.id DESC").
		Offset((page - 1) * size).Limit(size).Find(&items).Error
	return items, total, err
}

// CountTraces 命中检索条件的过程链总数（导出限额判定）
func (r *ObservabilityRepository) CountTraces(ctx context.Context, f TraceFilter) (int64, error) {
	var total int64
	err := r.filteredTraces(ctx, f).Count(&total).Error
	return total, err
}

// ListTracesForExport 按检索条件全量取过程链（导出，顺序与分页一致）
func (r *ObservabilityRepository) ListTracesForExport(ctx context.Context, f TraceFilter) ([]model.SysAiTrace, error) {
	var items []model.SysAiTrace
	err := r.filteredTraces(ctx, f).Select("sys_ai_trace.*").
		Order("sys_ai_trace.create_time DESC, sys_ai_trace.id DESC").Find(&items).Error
	return items, err
}

// GetTraceByTraceID 按 trace_id 查询过程链
func (r *ObservabilityRepository) GetTraceByTraceID(ctx context.Context, traceID string) (*model.SysAiTrace, error) {
	var item model.SysAiTrace
	err := r.db.WithContext(ctx).Where("trace_id = ?", traceID).First(&item).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	return &item, err
}

// ListTracesByConversation 会话全部过程链（含旁路，时间正序）
func (r *ObservabilityRepository) ListTracesByConversation(ctx context.Context, conversationID int64) ([]model.SysAiTrace, error) {
	var items []model.SysAiTrace
	err := r.db.WithContext(ctx).Where("conversation_id = ?", conversationID).
		Order("create_time ASC, id ASC").Find(&items).Error
	return items, err
}

// ConversationTitles 批量取会话标题（检索行透出会话归属）
func (r *ObservabilityRepository) ConversationTitles(ctx context.Context, ids []int64) (map[int64]string, error) {
	titles := make(map[int64]string, len(ids))
	if len(ids) == 0 {
		return titles, nil
	}
	var rows []struct {
		ID    int64  `gorm:"column:id"`
		Title string `gorm:"column:title"`
	}
	err := r.db.WithContext(ctx).Table("sys_ai_conversation").
		Select("id, title").Where("id IN ?", ids).Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	for _, row := range rows {
		titles[row.ID] = row.Title
	}
	return titles, nil
}

// ListMessagesAsc 会话消息按时间/id 正序取前 limit 条（过程链详情回放口径，与 python list_by_conversation 的 asc 一致）
func (r *ObservabilityRepository) ListMessagesAsc(ctx context.Context, conversationID int64, limit int) ([]model.SysAiMessage, error) {
	var items []model.SysAiMessage
	err := r.db.WithContext(ctx).
		Where("conversation_id = ? AND deleted = 0", conversationID).
		Order("create_time ASC, id ASC").Limit(limit).Find(&items).Error
	return items, err
}

// ListArtifactsByMessage 消息关联中间产物（create_time/id 倒序）
func (r *ObservabilityRepository) ListArtifactsByMessage(ctx context.Context, messageID int64) ([]model.SysAiArtifact, error) {
	var items []model.SysAiArtifact
	err := r.db.WithContext(ctx).
		Where("message_id = ?", messageID).
		Order("create_time DESC, id DESC").Find(&items).Error
	return items, err
}

// ==================== 总览统计 ====================

// TraceStatusCount 状态 → 过程链数
func (r *ObservabilityRepository) TraceStatusCount(ctx context.Context) (map[int]int64, error) {
	var rows []struct {
		Status int   `gorm:"column:status"`
		Total  int64 `gorm:"column:total"`
	}
	err := r.db.WithContext(ctx).Model(&model.SysAiTrace{}).
		Select("status, COUNT(*) AS total").Group("status").Scan(&rows).Error
	if err != nil {
		return nil, err
	}
	counts := make(map[int]int64, len(rows))
	for _, row := range rows {
		counts[row.Status] = row.Total
	}
	return counts, nil
}

// CountQuotaRejected 配额拒绝类过程链数
func (r *ObservabilityRepository) CountQuotaRejected(ctx context.Context) (int64, error) {
	var total int64
	err := r.db.WithContext(ctx).Model(&model.SysAiTrace{}).
		Where("error_type IN ?", quotaRejectErrorTypes).Count(&total).Error
	return total, err
}

// CountHighRisk 高风险调用数：推理步数超阈值，或存在"发起工具调用但调用失败/超时"的 LLM 调用
func (r *ObservabilityRepository) CountHighRisk(ctx context.Context) (int64, error) {
	risky := r.db.WithContext(ctx).Model(&model.SysAiLlmCall{}).
		Select("1").
		Where("sys_ai_llm_call.trace_id = sys_ai_trace.trace_id").
		Where("sys_ai_llm_call.tool_call IS NOT NULL AND sys_ai_llm_call.status <> 1")
	var total int64
	err := r.db.WithContext(ctx).Model(&model.SysAiTrace{}).
		Where("step_count >= ? OR EXISTS (?)", highRiskStepThreshold, risky).
		Count(&total).Error
	return total, err
}

// ==================== LLM 调用明细 ====================

// ListLLMCallsByTrace 单条过程链的调用明细（seq 正序）
func (r *ObservabilityRepository) ListLLMCallsByTrace(ctx context.Context, traceID string) ([]model.SysAiLlmCall, error) {
	var items []model.SysAiLlmCall
	err := r.db.WithContext(ctx).Where("trace_id = ?", traceID).Order("seq ASC").Find(&items).Error
	return items, err
}

// ListLLMCallsByTraces 批量查询多条过程链的调用明细，按 trace_id 分组（seq 正序，避免 N+1）
func (r *ObservabilityRepository) ListLLMCallsByTraces(ctx context.Context, traceIDs []string) (map[string][]model.SysAiLlmCall, error) {
	grouped := make(map[string][]model.SysAiLlmCall, len(traceIDs))
	if len(traceIDs) == 0 {
		return grouped, nil
	}
	var items []model.SysAiLlmCall
	if err := r.db.WithContext(ctx).Where("trace_id IN ?", traceIDs).Order("seq ASC").Find(&items).Error; err != nil {
		return nil, err
	}
	for _, item := range items {
		grouped[item.TraceID] = append(grouped[item.TraceID], item)
	}
	return grouped, nil
}

// ==================== 资源消耗 / 趋势聚合 ====================

// CostAggRow 按维度聚合的资源消耗行（Dimension 为分组值：model/agent_code/user_id 三者之一）
type CostAggRow struct {
	Dimension        *string `gorm:"column:dimension"`
	TraceCount       int64   `gorm:"column:trace_count"`
	TotalTokens      int64   `gorm:"column:total_tokens"`
	PromptTokens     int64   `gorm:"column:prompt_tokens"`
	CompletionTokens int64   `gorm:"column:completion_tokens"`
	CachedTokens     int64   `gorm:"column:cached_tokens"`
}

// CostTrendRow 按日 Token 消耗趋势行
type CostTrendRow struct {
	Date             string `gorm:"column:date"`
	TraceCount       int64  `gorm:"column:trace_count"`
	TotalTokens      int64  `gorm:"column:total_tokens"`
	PromptTokens     int64  `gorm:"column:prompt_tokens"`
	CompletionTokens int64  `gorm:"column:completion_tokens"`
	CachedTokens     int64  `gorm:"column:cached_tokens"`
}

const costMetricColumns = `COUNT(*) AS trace_count,
	COALESCE(SUM(sys_ai_trace.total_tokens), 0) AS total_tokens,
	COALESCE(SUM(sys_ai_trace.prompt_tokens), 0) AS prompt_tokens,
	COALESCE(SUM(sys_ai_trace.completion_tokens), 0) AS completion_tokens,
	COALESCE(SUM(sys_ai_trace.cached_tokens), 0) AS cached_tokens`

// costAggQuery 构造资源消耗聚合基础查询（每次调用新建，避免复用已执行语句）
func (r *ObservabilityRepository) costAggQuery(ctx context.Context, dimension string, start, end *time.Time) (*gorm.DB, string, error) {
	var dimExpr string
	db := r.db.WithContext(ctx).Model(&model.SysAiTrace{})
	switch dimension {
	case "model":
		dimExpr = "sys_ai_trace.model"
	case "agent":
		dimExpr = "sys_ai_trace.agent_code"
	case "user":
		dimExpr = "c.user_id"
		db = db.Joins("JOIN sys_ai_conversation AS c ON c.id = sys_ai_trace.conversation_id")
	default:
		return nil, "", errUnsupportedCostDimension
	}
	if start != nil {
		db = db.Where("sys_ai_trace.create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("sys_ai_trace.create_time <= ?", *end)
	}
	return db, dimExpr, nil
}

var errUnsupportedCostDimension = errors.New("unsupported cost aggregation dimension")

// PaginateCostAgg 资源消耗按维度分页聚合（分组总数 + 当前页）
func (r *ObservabilityRepository) PaginateCostAgg(
	ctx context.Context, dimension string, start, end *time.Time, page, size int,
) ([]CostAggRow, int64, error) {
	countDB, dimExpr, err := r.costAggQuery(ctx, dimension, start, end)
	if err != nil {
		return nil, 0, err
	}
	selectClause := dimExpr + " AS dimension, " + costMetricColumns
	countDB = countDB.Select(selectClause).Group("dimension")
	var total int64
	if err := r.db.WithContext(ctx).Table("(?) AS grouped", countDB).Count(&total).Error; err != nil {
		return nil, 0, err
	}

	rowDB, _, err := r.costAggQuery(ctx, dimension, start, end)
	if err != nil {
		return nil, 0, err
	}
	rows := make([]CostAggRow, 0, size)
	err = rowDB.Select(selectClause).Group("dimension").Order("dimension ASC").
		Offset((page - 1) * size).Limit(size).Scan(&rows).Error
	return rows, total, err
}

// CostTrendByDay 按日 Token 消耗趋势（与聚合口径一致）
func (r *ObservabilityRepository) CostTrendByDay(ctx context.Context, start, end *time.Time) ([]CostTrendRow, error) {
	db := r.db.WithContext(ctx).Model(&model.SysAiTrace{})
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var rows []CostTrendRow
	err := db.Select("DATE_FORMAT(sys_ai_trace.create_time, '%Y-%m-%d') AS date, " + costMetricColumns).
		Group("date").Order("date ASC").Scan(&rows).Error
	return rows, err
}

// TrendRow 性能趋势行
type TrendRow struct {
	Dimension       *string  `gorm:"column:dimension"`
	Date            string   `gorm:"column:date"`
	CallCount       int64    `gorm:"column:call_count"`
	SuccessCount    int64    `gorm:"column:success_count"`
	AvgFirstTokenMs *float64 `gorm:"column:avg_first_token_ms"`
	AvgDurationMs   *float64 `gorm:"column:avg_duration_ms"`
}

// PerformanceTrends 按维度 + 日期聚合调用量/成功率/平均延迟（首 Token 取成功调用口径）
func (r *ObservabilityRepository) PerformanceTrends(ctx context.Context, dimension string, start, end *time.Time) ([]TrendRow, error) {
	var dimExpr string
	switch dimension {
	case "model":
		dimExpr = "sys_ai_trace.model"
	case "agent":
		dimExpr = "sys_ai_trace.agent_code"
	default:
		return nil, errUnsupportedCostDimension
	}
	db := r.db.WithContext(ctx).Model(&model.SysAiTrace{})
	if start != nil {
		db = db.Where("create_time >= ?", *start)
	}
	if end != nil {
		db = db.Where("create_time <= ?", *end)
	}
	var rows []TrendRow
	err := db.Select(dimExpr + ` AS dimension,
			DATE_FORMAT(sys_ai_trace.create_time, '%Y-%m-%d') AS date,
			COUNT(*) AS call_count,
			COALESCE(SUM(CASE WHEN sys_ai_trace.status = 1 THEN 1 ELSE 0 END), 0) AS success_count,
			AVG(CASE WHEN sys_ai_trace.status = 1 THEN sys_ai_trace.first_token_ms END) AS avg_first_token_ms,
			AVG(sys_ai_trace.duration_ms) AS avg_duration_ms`).
		Group("dimension, date").Order("date ASC, dimension ASC").Scan(&rows).Error
	return rows, err
}
