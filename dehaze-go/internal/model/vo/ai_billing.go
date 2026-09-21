package vo

import (
	"encoding/json"
	"time"
)

// AI 计费视图对象（对齐 python app/models/schema/ai_billing*.py 的 camelCase 输出）。
// python 侧 Decimal 字段（余额/流水金额/单价）经 pydantic 序列化为字符串，此处统一用 string 承载；
// python 走 NonNullJSONResponse 递归剔除 null，故可选字段一律 omitempty 以对齐"字段缺失"语义。

// BillingBalanceVO 余额账户视图（权益缺失/停用时限额展示为 0）
type BillingBalanceVO struct {
	UserID         int64  `json:"userId"`
	CreditsBalance string `json:"creditsBalance"`
	ArrearsStatus  bool   `json:"arrearsStatus"`
	DailyUsed      int64  `json:"dailyUsed"`
	DailyLimit     int64  `json:"dailyLimit"`
	MonthlyUsed    int64  `json:"monthlyUsed"`
	MonthlyLimit   int64  `json:"monthlyLimit"`
}

// BillingRecordVO 计费明细
type BillingRecordVO struct {
	ID                int64     `json:"id"`
	UserID            int64     `json:"userId"`
	ConversationID    *int64    `json:"conversationId,omitempty"`
	MessageID         *int64    `json:"messageId,omitempty"`
	Model             string    `json:"model"`
	ActualModel       *string   `json:"actualModel,omitempty"`
	BillType          string    `json:"billType"`
	InputTokens       int       `json:"inputTokens"`
	CachedInputTokens int       `json:"cachedInputTokens"`
	OutputTokens      int       `json:"outputTokens"`
	Credits           int       `json:"credits"`
	CreditsSaved      int       `json:"creditsSaved"`
	ToolCredits       *int      `json:"toolCredits,omitempty"`
	QuotaConsumed     int       `json:"quotaConsumed"`
	PreDeduct         int       `json:"preDeduct"`
	RefundStatus      int       `json:"refundStatus"`
	CreateTime        time.Time `json:"createTime"`
}

// CreditLogVO 余额变动流水（amount/balanceAfter 为整数积分语义，输出为字符串对齐 python Decimal）
type CreditLogVO struct {
	ID           int64     `json:"id"`
	UserID       int64     `json:"userId"`
	Source       string    `json:"source"`
	Amount       string    `json:"amount"`
	BalanceAfter string    `json:"balanceAfter"`
	RelatedID    *int64    `json:"relatedId,omitempty"`
	Reason       *string   `json:"reason,omitempty"`
	OperatorID   *int64    `json:"operatorId,omitempty"`
	CreateTime   time.Time `json:"createTime"`
}

// BillingTrendPointVO 消耗趋势点
type BillingTrendPointVO struct {
	Date         string `json:"date"`
	Credits      int64  `json:"credits"`
	InputTokens  int64  `json:"inputTokens"`
	OutputTokens int64  `json:"outputTokens"`
}

// BillingModelDistVO 模型消耗分布项
type BillingModelDistVO struct {
	Model   string `json:"model"`
	Credits int64  `json:"credits"`
	Tokens  int64  `json:"tokens"`
}

// BillingSavingsVO 缓存节省汇总
type BillingSavingsVO struct {
	CachedInputTokens int64 `json:"cachedInputTokens"`
	CreditsSaved      int64 `json:"creditsSaved"`
}

// BillingSummaryVO 用户端消耗汇总（仅 chat 类记录，仅本人数据）
type BillingSummaryVO struct {
	TotalCredits      int64                 `json:"totalCredits"`
	InputTokens       int64                 `json:"inputTokens"`
	OutputTokens      int64                 `json:"outputTokens"`
	Trend             []BillingTrendPointVO `json:"trend"`
	ModelDistribution []BillingModelDistVO  `json:"modelDistribution"`
	Savings           BillingSavingsVO      `json:"savings"`
}

// BillingStatVO 管理员分维度统计（token 列仅 chat 类口径）
type BillingStatVO struct {
	Dimension         string  `json:"dimension"`
	TotalCredits      int64   `json:"totalCredits"`
	TotalInputTokens  int64   `json:"totalInputTokens"`
	TotalOutputTokens int64   `json:"totalOutputTokens"`
	CacheHitRate      float64 `json:"cacheHitRate"`
	CreditsSaved      int64   `json:"creditsSaved"`
	DegradationCount  int64   `json:"degradationCount"`
}

// BillVO 月结账单（balanceStart/balanceEnd 为 Decimal 语义，输出字符串）
type BillVO struct {
	UserID        int64            `json:"userId"`
	Month         string           `json:"month"`
	TotalConsume  int64            `json:"totalConsume"`
	TotalRecharge int64            `json:"totalRecharge"`
	TotalRefund   int64            `json:"totalRefund"`
	BalanceStart  string           `json:"balanceStart"`
	BalanceEnd    string           `json:"balanceEnd"`
	ItemSummary   map[string]int64 `json:"itemSummary"`
}

// RefundVO 退款申请
type RefundVO struct {
	ID          int64      `json:"id"`
	UserID      int64      `json:"userId"`
	BillingID   int64      `json:"billingId"`
	Amount      int        `json:"amount"`
	Reason      string     `json:"reason"`
	Status      int        `json:"status"`
	AuditorID   *int64     `json:"auditorId,omitempty"`
	AuditRemark *string    `json:"auditRemark,omitempty"`
	CreateTime  time.Time  `json:"createTime"`
	UpdateTime  *time.Time `json:"updateTime,omitempty"`
}

// AnomalyVO 计费异常事件
type AnomalyVO struct {
	ID          int64     `json:"id"`
	UserID      int64     `json:"userId"`
	BillingID   *int64    `json:"billingId,omitempty"`
	AnomalyType string    `json:"anomalyType"`
	Detail      string    `json:"detail"`
	Status      int       `json:"status"`
	TriggerAt   time.Time `json:"triggerAt"`
	CreateTime  time.Time `json:"createTime"`
}

// ModelCostDetailVO 成本档位明细（unitPrice 为 Decimal 语义，输出字符串）
type ModelCostDetailVO struct {
	ID        int64  `json:"id"`
	PriceID   int64  `json:"priceId"`
	TokenType string `json:"tokenType"`
	TimeSlot  string `json:"timeSlot"`
	MinTokens int64  `json:"minTokens"`
	MaxTokens *int64 `json:"maxTokens,omitempty"`
	UnitPrice string `json:"unitPrice"`
}

// ModelCostVO 成本单价版本
type ModelCostVO struct {
	ID            int64               `json:"id"`
	ModelID       string              `json:"modelId"`
	ProviderID    int64               `json:"providerId"`
	PriceVersion  int                 `json:"priceVersion"`
	Currency      string              `json:"currency"`
	EffectiveFrom time.Time           `json:"effectiveFrom"`
	EffectiveTo   *time.Time          `json:"effectiveTo,omitempty"`
	Status        int                 `json:"status"`
	Details       []ModelCostDetailVO `json:"details"`
	CreateTime    *time.Time          `json:"createTime,omitempty"`
	UpdateTime    *time.Time          `json:"updateTime,omitempty"`
}

// CostStatVO 成本-利润统计项（overall/ai 口径含收入与毛利；model/provider 分组仅返回维度值与成本）
type CostStatVO struct {
	Dimension  *string  `json:"dimension,omitempty"`
	Revenue    *float64 `json:"revenue,omitempty"`
	Cost       float64  `json:"cost"`
	Profit     *float64 `json:"profit,omitempty"`
	ProfitRate *float64 `json:"profitRate,omitempty"`
	Metric     *string  `json:"metric,omitempty"`
}

// ==================== 可观测性 ====================

// TraceVO 过程链检索行
type TraceVO struct {
	TraceID           string    `json:"traceId"`
	ConversationID    int64     `json:"conversationId"`
	ConversationTitle *string   `json:"conversationTitle,omitempty"`
	MessageID         *int64    `json:"messageId,omitempty"`
	AgentCode         *string   `json:"agentCode,omitempty"`
	TraceType         string    `json:"traceType"`
	Model             *string   `json:"model,omitempty"`
	Status            int       `json:"status"`
	ErrorType         *string   `json:"errorType,omitempty"`
	DurationMs        int       `json:"durationMs"`
	FirstTokenMs      *int      `json:"firstTokenMs,omitempty"`
	LlmCallCount      int       `json:"llmCallCount"`
	TotalTokens       int       `json:"totalTokens"`
	PromptTokens      int       `json:"promptTokens"`
	CompletionTokens  int       `json:"completionTokens"`
	CachedTokens      int       `json:"cachedTokens"`
	StepCount         int       `json:"stepCount"`
	CreateTime        time.Time `json:"createTime"`
}

// LlmCallVO LLM 调用明细
type LlmCallVO struct {
	Seq              int             `json:"seq"`
	StepPosition     *int            `json:"stepPosition,omitempty"`
	Model            *string         `json:"model,omitempty"`
	Status           int             `json:"status"`
	ErrorType        *string         `json:"errorType,omitempty"`
	DurationMs       int             `json:"durationMs"`
	FirstTokenMs     *int            `json:"firstTokenMs,omitempty"`
	PromptTokens     int             `json:"promptTokens"`
	CompletionTokens int             `json:"completionTokens"`
	CachedTokens     int             `json:"cachedTokens"`
	ToolCall         json.RawMessage `json:"toolCall,omitempty"`
	InputSnapshot    json.RawMessage `json:"inputSnapshot,omitempty"`
	OutputSnapshot   json.RawMessage `json:"outputSnapshot,omitempty"`
	Attempts         json.RawMessage `json:"attempts,omitempty"`
	StartTime        *time.Time      `json:"startTime,omitempty"`
	RawRequest       json.RawMessage `json:"rawRequest,omitempty"`
	RawResponse      json.RawMessage `json:"rawResponse,omitempty"`
	CreateTime       time.Time       `json:"createTime"`
}

// TraceBillingVO 过程链关联计费记录
type TraceBillingVO struct {
	BillType          *string   `json:"billType,omitempty"`
	Model             *string   `json:"model,omitempty"`
	ActualModel       *string   `json:"actualModel,omitempty"`
	ProviderID        *int64    `json:"providerId,omitempty"`
	InputTokens       int       `json:"inputTokens"`
	OutputTokens      int       `json:"outputTokens"`
	CachedInputTokens int       `json:"cachedInputTokens"`
	Credits           int       `json:"credits"`
	CreditsSaved      int       `json:"creditsSaved"`
	ErrorCode         *string   `json:"errorCode,omitempty"`
	LatencyMs         *int      `json:"latencyMs,omitempty"`
	RequestID         *string   `json:"requestId,omitempty"`
	CreateTime        time.Time `json:"createTime"`
}

// TraceArtifactVO 过程链关联中间产物
type TraceArtifactVO struct {
	ID         int64           `json:"id"`
	Type       *string         `json:"type,omitempty"`
	Summary    json.RawMessage `json:"summary,omitempty"`
	RefType    *string         `json:"refType,omitempty"`
	RefID      *int64          `json:"refId,omitempty"`
	CreateTime time.Time       `json:"createTime"`
}

// TraceMessageVO 过程链所属会话消息
type TraceMessageVO struct {
	ID              int64     `json:"id"`
	ConversationID  int64     `json:"conversationId"`
	ParentMessageID *int64    `json:"parentMessageId,omitempty"`
	Role            string    `json:"role"`
	Content         *string   `json:"content,omitempty"`
	Status          int       `json:"status"`
	Model           *string   `json:"model,omitempty"`
	InputTokens     int       `json:"inputTokens"`
	OutputTokens    int       `json:"outputTokens"`
	CreateTime      time.Time `json:"createTime"`
}

// TraceDetailVO 过程链详情（上下文快照 + 调用回放 + 推理步骤 + 计费 + 产物）
type TraceDetailVO struct {
	TraceVO
	ContextSnapshot json.RawMessage   `json:"contextSnapshot,omitempty"`
	LlmCalls        []LlmCallVO       `json:"llmCalls"`
	Thoughts        []AgentThoughtVO  `json:"thoughts"`
	Messages        []TraceMessageVO  `json:"messages"`
	Billing         []TraceBillingVO  `json:"billing"`
	Artifacts       []TraceArtifactVO `json:"artifacts"`
	ErrorDetail     json.RawMessage   `json:"errorDetail,omitempty"`
}

// AgentThoughtVO 推理步骤（对齐 python AgentThoughtResult）
type AgentThoughtVO struct {
	ID             int64           `json:"id"`
	MessageID      int64           `json:"messageId"`
	ConversationID int64           `json:"conversationId"`
	Position       int             `json:"position"`
	AgentCode      *string         `json:"agentCode,omitempty"`
	IsSubagent     int             `json:"isSubagent"`
	Thought        *string         `json:"thought,omitempty"`
	Tool           *string         `json:"tool,omitempty"`
	ToolInput      json.RawMessage `json:"toolInput,omitempty"`
	Observation    *string         `json:"observation,omitempty"`
	Status         int             `json:"status"`
	LatencyMs      int             `json:"latencyMs"`
	Error          *string         `json:"error,omitempty"`
	CreateTime     time.Time       `json:"createTime"`
}

// ObservabilitySummaryVO 异常总览统计
type ObservabilitySummaryVO struct {
	Total            int64 `json:"total"`
	SuccessCount     int64 `json:"successCount"`
	FailedCount      int64 `json:"failedCount"`
	InterruptedCount int64 `json:"interruptedCount"`
	TimeoutCount     int64 `json:"timeoutCount"`
	QuotaRejected    int64 `json:"quotaRejected"`
	HighRiskCalls    int64 `json:"highRiskCalls"`
}

// CostItemVO 资源消耗聚合项（按模型/智能体/用户维度）
type CostItemVO struct {
	Model            *string `json:"model,omitempty"`
	AgentCode        *string `json:"agentCode,omitempty"`
	UserID           *int64  `json:"userId,omitempty"`
	TraceCount       int64   `json:"traceCount"`
	TotalTokens      int64   `json:"totalTokens"`
	PromptTokens     int64   `json:"promptTokens"`
	CompletionTokens int64   `json:"completionTokens"`
	CachedTokens     int64   `json:"cachedTokens"`
}

// CostTrendItemVO 按日 Token 消耗趋势项
type CostTrendItemVO struct {
	Date             string `json:"date"`
	TraceCount       int64  `json:"traceCount"`
	TotalTokens      int64  `json:"totalTokens"`
	PromptTokens     int64  `json:"promptTokens"`
	CompletionTokens int64  `json:"completionTokens"`
	CachedTokens     int64  `json:"cachedTokens"`
}

// ObservabilityCostsVO 资源消耗聚合结果
type ObservabilityCostsVO struct {
	Items []CostItemVO      `json:"items"`
	Total int64             `json:"total"`
	Trend []CostTrendItemVO `json:"trend"`
}

// TrendVO 性能趋势项（首 Token 延迟取成功调用口径）
type TrendVO struct {
	Model           *string  `json:"model,omitempty"`
	AgentCode       *string  `json:"agentCode,omitempty"`
	Date            string   `json:"date"`
	CallCount       int64    `json:"callCount"`
	SuccessCount    int64    `json:"successCount"`
	SuccessRate     float64  `json:"successRate"`
	AvgFirstTokenMs *float64 `json:"avgFirstTokenMs,omitempty"`
	AvgDurationMs   *float64 `json:"avgDurationMs,omitempty"`
}

// ==================== 会话审计时间线 ====================

// TimelineMessageVO 时间线消息
type TimelineMessageVO struct {
	ID           int64      `json:"id"`
	Role         string     `json:"role"`
	Content      *string    `json:"content,omitempty"`
	Status       int        `json:"status"`
	Model        *string    `json:"model,omitempty"`
	InputTokens  int        `json:"inputTokens"`
	OutputTokens int        `json:"outputTokens"`
	CreateTime   *time.Time `json:"createTime,omitempty"`
}

// TimelineEventVO 轮内事件（按 ts 交织排序，字段按 kind 取用）
type TimelineEventVO struct {
	Kind             string             `json:"kind"`
	Ts               *time.Time         `json:"ts,omitempty"`
	Message          *TimelineMessageVO `json:"message,omitempty"`
	Snapshot         json.RawMessage    `json:"snapshot,omitempty"`
	Seq              *int               `json:"seq,omitempty"`
	Model            *string            `json:"model,omitempty"`
	Status           *int               `json:"status,omitempty"`
	DurationMs       *int               `json:"durationMs,omitempty"`
	FirstTokenMs     *int               `json:"firstTokenMs,omitempty"`
	PromptTokens     *int               `json:"promptTokens,omitempty"`
	CompletionTokens *int               `json:"completionTokens,omitempty"`
	CachedTokens     *int               `json:"cachedTokens,omitempty"`
	ToolCall         json.RawMessage    `json:"toolCall,omitempty"`
	Attempts         json.RawMessage    `json:"attempts,omitempty"`
	RawRequest       json.RawMessage    `json:"rawRequest,omitempty"`
	RawResponse      json.RawMessage    `json:"rawResponse,omitempty"`
	Summary          json.RawMessage    `json:"summary,omitempty"`
	Position         *int               `json:"position,omitempty"`
	Tool             *string            `json:"tool,omitempty"`
	Thought          *string            `json:"thought,omitempty"`
	ToolInput        json.RawMessage    `json:"toolInput,omitempty"`
	Observation      *string            `json:"observation,omitempty"`
	LatencyMs        *int               `json:"latencyMs,omitempty"`
	AgentCode        *string            `json:"agentCode,omitempty"`
	IsSubagent       *int               `json:"isSubagent,omitempty"`
	Event            *string            `json:"event,omitempty"`
	Detail           json.RawMessage    `json:"detail,omitempty"`
	BillType         *string            `json:"billType,omitempty"`
	Credits          *int               `json:"credits,omitempty"`
	Tokens           json.RawMessage    `json:"tokens,omitempty"`
}

// TimelineTraceVO 轮次内的过程链（主对话 + 旁路）
type TimelineTraceVO struct {
	TraceID     string            `json:"traceId"`
	TraceType   string            `json:"traceType"`
	Status      int               `json:"status"`
	ErrorType   *string           `json:"errorType,omitempty"`
	ErrorDetail json.RawMessage   `json:"errorDetail,omitempty"`
	Model       *string           `json:"model,omitempty"`
	DurationMs  int               `json:"durationMs"`
	CreateTime  time.Time         `json:"createTime"`
	Events      []TimelineEventVO `json:"events"`
}

// TimelineRoundVO 会话轮次（user → assistant 配对 + 该轮过程链）
type TimelineRoundVO struct {
	UserMessage      *TimelineMessageVO `json:"userMessage,omitempty"`
	AssistantMessage *TimelineMessageVO `json:"assistantMessage,omitempty"`
	Traces           []TimelineTraceVO  `json:"traces"`
}

// TimelineConversationVO 时间线会话元信息
type TimelineConversationVO struct {
	ID         int64      `json:"id"`
	Title      string     `json:"title"`
	UserID     int64      `json:"userId"`
	AgentCode  *string    `json:"agentCode,omitempty"`
	CreateTime *time.Time `json:"createTime,omitempty"`
}

// TimelineVO 会话审计时间线
type TimelineVO struct {
	Conversation TimelineConversationVO `json:"conversation"`
	Rounds       []TimelineRoundVO      `json:"rounds"`
}
