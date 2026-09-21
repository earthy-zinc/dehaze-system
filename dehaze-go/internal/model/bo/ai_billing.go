package bo

import "time"

// ==================== AI 计费查询与表单（对齐 python app/models/schema/ai_billing*.py） ====================
//
// 分页字段一律不参与 gin 表单绑定（`form:"-"`），由各 handler 调 parsePaginationWithSize 统一解析：
// python 的 pageNum/pageSize 是 Query(default, ge=1, le=100)，非法值报 A0400；若交给 ShouldBindQuery，
// `pageNum=abc` 这类解析失败会走 error_handler 兜底分支报 B0001，且 `pageNum=0` 无从区分"未传"与"显式 0"，
// 会被静默当成缺省 —— 两条都与 python 分叉。

// BillingRecordQuery 计费明细查询（用户端；userId 供管理端下钻，需 ai:billing:stat）
type BillingRecordQuery struct {
	PageNum        int    `form:"-" json:"pageNum"`
	PageSize       int    `form:"-" json:"pageSize"`
	UserID         *int64 `form:"userId" json:"userId"`
	ConversationID *int64 `form:"conversationId" json:"conversationId"`
	BillType       string `form:"billType" json:"billType"`
	ModelID        string `form:"modelId" json:"modelId"`
	DateStart      string `form:"dateStart" json:"dateStart"`
	DateEnd        string `form:"dateEnd" json:"dateEnd"`
}

// CreditLogQuery 余额流水查询
type CreditLogQuery struct {
	PageNum   int    `form:"-" json:"pageNum"`
	PageSize  int    `form:"-" json:"pageSize"`
	UserID    *int64 `form:"userId" json:"userId"`
	Source    string `form:"source" json:"source"`
	DateStart string `form:"dateStart" json:"dateStart"`
	DateEnd   string `form:"dateEnd" json:"dateEnd"`
}

// BillingStatQuery 管理员计费统计查询（groupBy: user/model/billType/day）
type BillingStatQuery struct {
	UserID    *int64 `form:"userId" json:"userId"`
	ModelID   string `form:"modelId" json:"modelId"`
	BillType  string `form:"billType" json:"billType"`
	DateStart string `form:"dateStart" json:"dateStart"`
	DateEnd   string `form:"dateEnd" json:"dateEnd"`
	GroupBy   string `form:"groupBy" json:"groupBy"`
}

// RefundQuery 退款申请列表查询
type RefundQuery struct {
	PageNum   int    `form:"-" json:"pageNum"`
	PageSize  int    `form:"-" json:"pageSize"`
	UserID    *int64 `form:"userId" json:"userId"`
	Status    *int   `form:"status" json:"status"`
	DateStart string `form:"dateStart" json:"dateStart"`
	DateEnd   string `form:"dateEnd" json:"dateEnd"`
}

// AnomalyQuery 异常计费记录查询
type AnomalyQuery struct {
	PageNum     int    `form:"-" json:"pageNum"`
	PageSize    int    `form:"-" json:"pageSize"`
	UserID      *int64 `form:"userId" json:"userId"`
	AnomalyType string `form:"anomalyType" json:"anomalyType"`
	Status      *int   `form:"status" json:"status"`
	DateStart   string `form:"dateStart" json:"dateStart"`
	DateEnd     string `form:"dateEnd" json:"dateEnd"`
}

// BillingRefundApplyForm 用户退款申请（amount <= 0 由 service 按 python 语义报 A0400，故不设 required）
type BillingRefundApplyForm struct {
	BillingID int64  `json:"billingId" binding:"required"`
	Amount    int    `json:"amount"`
	Reason    string `json:"reason" binding:"required"`
}

// BillingRefundAuditForm 退款审核（approved 缺失按 python 必填语义报参数错误，用指针区分"未传"与 false）
type BillingRefundAuditForm struct {
	Approved    *bool   `json:"approved" binding:"required"`
	AuditRemark *string `json:"auditRemark"`
}

// CreditAdjustForm 管理员手动调整积分（amount == 0 由 service 按 python 语义报 A0400）
type CreditAdjustForm struct {
	UserID int64  `json:"userId" binding:"required"`
	Amount int    `json:"amount"`
	Reason string `json:"reason" binding:"required"`
}

// ==================== 成本管理 ====================

// ModelCostQuery 成本单价列表查询
type ModelCostQuery struct {
	PageNum    int    `form:"-" json:"pageNum"`
	PageSize   int    `form:"-" json:"pageSize"`
	Keyword    string `form:"keyword" json:"keyword"`
	ModelID    string `form:"modelId" json:"modelId"`
	ProviderID *int64 `form:"providerId" json:"providerId"`
}

// ModelCostDetailForm 成本档位明细表单
type ModelCostDetailForm struct {
	TokenType string  `json:"tokenType" binding:"required"`
	TimeSlot  string  `json:"timeSlot" binding:"required"`
	MinTokens int64   `json:"minTokens"`
	MaxTokens *int64  `json:"maxTokens"`
	UnitPrice float64 `json:"unitPrice"`
}

// ModelCostCreateForm 新增成本单价（同模型同供应商生成新价格版本）
type ModelCostCreateForm struct {
	ModelID       string                `json:"modelId" binding:"required"`
	ProviderID    int64                 `json:"providerId" binding:"required"`
	Currency      string                `json:"currency"`
	EffectiveFrom *time.Time            `json:"effectiveFrom"`
	EffectiveTo   *time.Time            `json:"effectiveTo"`
	Status        *int                  `json:"status"`
	Details       []ModelCostDetailForm `json:"details"`
}

// ModelCostUpdateForm 更新成本单价（仅版本主表字段）
type ModelCostUpdateForm struct {
	Currency      *string    `json:"currency"`
	EffectiveFrom *time.Time `json:"effectiveFrom"`
	EffectiveTo   *time.Time `json:"effectiveTo"`
	Status        *int       `json:"status"`
}

// CostStatQuery 成本-利润统计查询（groupBy: overall/model/provider）
type CostStatQuery struct {
	StartTime  string `form:"startTime" json:"startTime"`
	EndTime    string `form:"endTime" json:"endTime"`
	GroupBy    string `form:"groupBy" json:"groupBy"`
	ModelID    string `form:"modelId" json:"modelId"`
	ProviderID *int64 `form:"providerId" json:"providerId"`
}

// ReconcileImportForm 供应商账单导入
type ReconcileImportForm struct {
	Content   string  `json:"content" binding:"required"`
	StartTime *string `json:"startTime"`
	EndTime   *string `json:"endTime"`
}

// ==================== 可观测性（F-M08-013） ====================

// TracePageQuery 过程链检索查询
type TracePageQuery struct {
	PageNum        int    `form:"-" json:"pageNum"`
	PageSize       int    `form:"-" json:"pageSize"`
	ConversationID *int64 `form:"conversationId" json:"conversationId"`
	UserID         *int64 `form:"userId" json:"userId"`
	Status         *int   `form:"status" json:"status"`
	AgentCode      string `form:"agentCode" json:"agentCode"`
	Model          string `form:"model" json:"model"`
	ErrorType      string `form:"errorType" json:"errorType"`
	Keyword        string `form:"keyword" json:"keyword"`
	Capability     string `form:"capability" json:"capability"`
	StartTime      string `form:"startTime" json:"startTime"`
	EndTime        string `form:"endTime" json:"endTime"`
}

// CostsQuery 资源消耗聚合查询
type CostsQuery struct {
	PageNum   int    `form:"-" json:"pageNum"`
	PageSize  int    `form:"-" json:"pageSize"`
	Dimension string `form:"dimension" json:"dimension"`
	StartTime string `form:"startTime" json:"startTime"`
	EndTime   string `form:"endTime" json:"endTime"`
}

// TrendsQuery 性能趋势查询
type TrendsQuery struct {
	Dimension string `form:"dimension" json:"dimension"`
	StartTime string `form:"startTime" json:"startTime"`
	EndTime   string `form:"endTime" json:"endTime"`
}
