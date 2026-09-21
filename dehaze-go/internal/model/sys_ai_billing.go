package model

import "time"

// AI 计费域实体（基础模块-AI计费管理）。
// 只追加类表（billing / credit_log / refund / anomaly）无逻辑删除；成本配置类表（model_cost 及其明细）
// 使用方案 A 软删（deleted = id），查询侧一律显式 deleted = 0。

// SysAiBilling AI 计费记录（每次 AI 能力调用的 Token/积分消耗明细）
type SysAiBilling struct {
	ID                int64   `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID            int64   `gorm:"column:user_id" json:"userId"`
	ConversationID    *int64  `gorm:"column:conversation_id" json:"conversationId"`
	MessageID         *int64  `gorm:"column:message_id" json:"messageId"`
	RequestID         *string `gorm:"column:request_id" json:"requestId"`
	ProviderID        *int64  `gorm:"column:provider_id" json:"providerId"`
	Model             string  `gorm:"column:model" json:"model"`
	ActualModel       *string `gorm:"column:actual_model" json:"actualModel"`
	ErrorCode         *string `gorm:"column:error_code" json:"errorCode"`
	LatencyMs         *int    `gorm:"column:latency_ms" json:"latencyMs"`
	BillType          string  `gorm:"column:bill_type" json:"billType"`
	InputTokens       int     `gorm:"column:input_tokens" json:"inputTokens"`
	CachedInputTokens int     `gorm:"column:cached_input_tokens" json:"cachedInputTokens"`
	OutputTokens      int     `gorm:"column:output_tokens" json:"outputTokens"`
	Credits           int     `gorm:"column:credits" json:"credits"`
	CreditsSaved      int     `gorm:"column:credits_saved" json:"creditsSaved"`
	ToolCredits       *int    `gorm:"column:tool_credits" json:"toolCredits"`
	QuotaConsumed     int     `gorm:"column:quota_consumed" json:"quotaConsumed"`
	PreDeduct         int     `gorm:"column:pre_deduct" json:"preDeduct"`
	// 成本（元，成本线异步回填）：仅管理端聚合口径使用，绝不随计费记录下发（json:"-"）
	Cost       *float64  `gorm:"column:cost" json:"-"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiBilling) TableName() string { return "sys_ai_billing" }

// SysAiCreditLog 积分余额变动流水（只追加，表内 deleted 列不参与查询过滤）
type SysAiCreditLog struct {
	ID           int64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID       int64     `gorm:"column:user_id" json:"userId"`
	Source       string    `gorm:"column:source" json:"source"`
	Amount       int64     `gorm:"column:amount" json:"amount"`
	BalanceAfter int64     `gorm:"column:balance_after" json:"balanceAfter"`
	RelatedID    *int64    `gorm:"column:related_id" json:"relatedId"`
	Reason       *string   `gorm:"column:reason" json:"reason"`
	OperatorID   *int64    `gorm:"column:operator_id" json:"operatorId"`
	CreateTime   time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiCreditLog) TableName() string { return "sys_ai_credit_log" }

// SysAiRefund AI 积分误扣退款申请（1待审核 → 2已通过 / 3已驳回）
type SysAiRefund struct {
	ID          int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID      int64      `gorm:"column:user_id" json:"userId"`
	BillingID   int64      `gorm:"column:billing_id" json:"billingId"`
	Amount      int        `gorm:"column:amount" json:"amount"`
	Reason      string     `gorm:"column:reason" json:"reason"`
	Status      int        `gorm:"column:status" json:"status"`
	AuditorID   *int64     `gorm:"column:auditor_id" json:"auditorId"`
	AuditRemark *string    `gorm:"column:audit_remark" json:"auditRemark"`
	CreateBy    *int64     `gorm:"column:create_by" json:"-"`
	CreateTime  time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime  *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiRefund) TableName() string { return "sys_ai_refund" }

// SysAiBillingAnomaly 计费异常事件（四类规则命中结果，只追加）
type SysAiBillingAnomaly struct {
	ID          int64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	UserID      int64     `gorm:"column:user_id" json:"userId"`
	BillingID   *int64    `gorm:"column:billing_id" json:"billingId"`
	AnomalyType string    `gorm:"column:anomaly_type" json:"anomalyType"`
	Detail      string    `gorm:"column:detail" json:"detail"`
	Status      int       `gorm:"column:status" json:"status"`
	TriggerAt   time.Time `gorm:"column:trigger_at" json:"triggerAt"`
	CreateTime  time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiBillingAnomaly) TableName() string { return "sys_ai_billing_anomaly" }

// SysAiModelCost 模型成本单价版本（供应商采购价，软删）
type SysAiModelCost struct {
	ID            int64      `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	ModelID       string     `gorm:"column:model_id" json:"modelId"`
	ProviderID    int64      `gorm:"column:provider_id" json:"providerId"`
	PriceVersion  int        `gorm:"column:price_version" json:"priceVersion"`
	Currency      string     `gorm:"column:currency" json:"currency"`
	EffectiveFrom time.Time  `gorm:"column:effective_from" json:"effectiveFrom"`
	EffectiveTo   *time.Time `gorm:"column:effective_to" json:"effectiveTo"`
	Status        int        `gorm:"column:status" json:"status"`
	Deleted       int64      `gorm:"column:deleted" json:"-"`
	CreateBy      *int64     `gorm:"column:create_by" json:"-"`
	UpdateBy      *int64     `gorm:"column:update_by" json:"-"`
	CreateTime    time.Time  `gorm:"column:create_time;autoCreateTime" json:"createTime"`
	UpdateTime    *time.Time `gorm:"column:update_time;autoUpdateTime" json:"updateTime"`
}

func (SysAiModelCost) TableName() string { return "sys_ai_model_cost" }

// SysAiModelCostDetail 成本单价档位明细（token 类型 × 上下文分段 × 时段）
type SysAiModelCostDetail struct {
	ID         int64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	PriceID    int64     `gorm:"column:price_id" json:"priceId"`
	TokenType  string    `gorm:"column:token_type" json:"tokenType"`
	TimeSlot   string    `gorm:"column:time_slot" json:"timeSlot"`
	MinTokens  int64     `gorm:"column:min_tokens" json:"minTokens"`
	MaxTokens  *int64    `gorm:"column:max_tokens" json:"maxTokens"`
	UnitPrice  float64   `gorm:"column:unit_price" json:"-"`
	Deleted    int64     `gorm:"column:deleted" json:"-"`
	CreateTime time.Time `gorm:"column:create_time;autoCreateTime" json:"createTime"`
}

func (SysAiModelCostDetail) TableName() string { return "sys_ai_model_cost_detail" }
