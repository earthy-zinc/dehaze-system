package bo

import (
	"encoding/json"
	"time"
)

// ==================== 分页查询 ====================

// AiPageQuery 通用分页查询（pageNum/pageSize，与 python BasePageQuery 一致）。
//
// 分页字段刻意不参与 gin 绑定（`form:"-"`）：绑定层会把非数字兜底成 B0001，且越界值无法表达为
// A0400。分页一律由 handler 用 parsePaginationWithSize/parsePaginationNamed 校验后回填，
// 使"越界 A0400"只有一处实现（见 internal/api/sys_feedback.go）。
type AiPageQuery struct {
	PageNum  int `form:"-" json:"pageNum"`
	PageSize int `form:"-" json:"pageSize"`
}

// ==================== 模型注册表 ====================

// AiModelCreateForm 新增模型表单
type AiModelCreateForm struct {
	ProviderID            int64           `json:"providerId" binding:"required"`
	ModelID               string          `json:"modelId" binding:"required,max=64"`
	ModelType             string          `json:"modelType"`
	Dimension             *int64          `json:"dimension"`
	DisplayName           string          `json:"displayName" binding:"required,max=128"`
	MaxContextTokens      *int            `json:"maxContextTokens"`
	MaxOutputTokens       *int            `json:"maxOutputTokens"`
	SupportsMultimodal    *bool           `json:"supportsMultimodal"`
	SupportsToolCall      *bool           `json:"supportsToolCall"`
	SupportsStreaming     *bool           `json:"supportsStreaming"`
	SupportsPromptCache   *bool           `json:"supportsPromptCache"`
	SupportsStructuredOut *bool           `json:"supportsStructuredOutput"`
	ExtraRequestParams    json.RawMessage `json:"extraRequestParams"`
	FallbackModelID       *int64          `json:"fallbackModelId"`
	PromptCachePrefixLen  *int            `json:"promptCachePrefixLen"`
	Status                *int8           `json:"status"`
	VipLevel              *int8           `json:"vipLevel"`
}

// AiModelUpdateForm 更新模型表单（仅传需要变更的字段；modelType/dimension 创建后不可改）
type AiModelUpdateForm struct {
	ProviderID            *int64          `json:"providerId"`
	ModelType             *string         `json:"modelType"`
	Dimension             *int64          `json:"dimension"`
	DisplayName           *string         `json:"displayName"`
	MaxContextTokens      *int            `json:"maxContextTokens"`
	MaxOutputTokens       *int            `json:"maxOutputTokens"`
	SupportsMultimodal    *bool           `json:"supportsMultimodal"`
	SupportsToolCall      *bool           `json:"supportsToolCall"`
	SupportsStreaming     *bool           `json:"supportsStreaming"`
	SupportsPromptCache   *bool           `json:"supportsPromptCache"`
	SupportsStructuredOut *bool           `json:"supportsStructuredOutput"`
	ExtraRequestParams    json.RawMessage `json:"extraRequestParams"`
	// FallbackModelID：nil=未传；"null"=显式清空；数字=设置（json.RawMessage 保留显式 null 语义）
	FallbackModelID      json.RawMessage `json:"fallbackModelId"`
	PromptCachePrefixLen *int            `json:"promptCachePrefixLen"`
	Status               *int8           `json:"status"`
	VipLevel             *int8           `json:"vipLevel"`
}

// AiModelQuery 模型分页查询
type AiModelQuery struct {
	AiPageQuery
	Keyword   string `form:"keyword"`
	ModelType string `form:"modelType"`
}

// ==================== 模型用户售价 ====================

// ModelPriceDetailForm 价格档位明细表单
type ModelPriceDetailForm struct {
	TokenType string  `json:"tokenType" binding:"required"`
	TimeSlot  string  `json:"timeSlot" binding:"required"`
	MinTokens int64   `json:"minTokens"`
	MaxTokens *int64  `json:"maxTokens"`
	UnitPrice float64 `json:"unitPrice"`
}

// ModelPriceCreateForm 新增价格版本表单（model_id 由路径参数指定）
type ModelPriceCreateForm struct {
	ProviderID    int64                  `json:"providerId" binding:"required"`
	Unit          string                 `json:"unit"`
	EffectiveFrom *time.Time             `json:"effectiveFrom"`
	EffectiveTo   *time.Time             `json:"effectiveTo"`
	Status        *int8                  `json:"status"`
	Details       []ModelPriceDetailForm `json:"details"`
}

// ModelPriceUpdateForm 更新价格版本表单（仅主表字段）
type ModelPriceUpdateForm struct {
	Unit          *string    `json:"unit"`
	EffectiveFrom *time.Time `json:"effectiveFrom"`
	EffectiveTo   *time.Time `json:"effectiveTo"`
	Status        *int8      `json:"status"`
}

// ModelPriceQuery 价格版本分页查询（page/size，SDK 口径）
type ModelPriceQuery struct {
	Page       int    `form:"page"`
	Size       int    `form:"size"`
	ProviderID *int64 `form:"providerId"`
}

// ==================== 供应商 ====================

// ProviderCreateForm 新增供应商表单
type ProviderCreateForm struct {
	ProviderCode        string          `json:"providerCode" binding:"required,max=32"`
	DisplayName         string          `json:"displayName" binding:"required,max=128"`
	ApiBaseUrl          string          `json:"apiBaseUrl" binding:"required,max=512"`
	ProtocolType        string          `json:"protocolType"`
	AuthType            string          `json:"authType"`
	DefaultHeaders      json.RawMessage `json:"defaultHeaders"`
	SortOrder           *int            `json:"sortOrder"`
	HealthCheckEnabled  *int8           `json:"healthCheckEnabled"`
	UserIdentityForward json.RawMessage `json:"userIdentityForward"`
	Remark              json.RawMessage `json:"remark"`
	Status              *int8           `json:"status"`
}

// ProviderUpdateForm 更新供应商表单
type ProviderUpdateForm struct {
	DisplayName        *string         `json:"displayName"`
	ApiBaseUrl         *string         `json:"apiBaseUrl"`
	ProtocolType       *string         `json:"protocolType"`
	AuthType           *string         `json:"authType"`
	DefaultHeaders     json.RawMessage `json:"defaultHeaders"`
	SortOrder          *int            `json:"sortOrder"`
	HealthCheckEnabled *int8           `json:"healthCheckEnabled"`
	// UserIdentityForward / Remark：nil=未传；"null"=显式清空
	UserIdentityForward json.RawMessage `json:"userIdentityForward"`
	Remark              json.RawMessage `json:"remark"`
	Status              *int8           `json:"status"`
}

// ProviderQuery 供应商分页查询
type ProviderQuery struct {
	AiPageQuery
	Keyword string `form:"keyword"`
}

// ProviderKeyCreateForm 新增 API Key 表单
type ProviderKeyCreateForm struct {
	Name       string     `json:"name" binding:"required,max=128"`
	Key        string     `json:"key" binding:"required"`
	Priority   *int       `json:"priority"`
	Weight     *int       `json:"weight"`
	DailyQuota *int       `json:"dailyQuota"`
	RpmLimit   *int       `json:"rpmLimit"`
	ExpiresAt  *time.Time `json:"expiresAt"`
	Status     *int8      `json:"status"`
}

// ProviderKeyUpdateForm 更新 API Key 表单（Key 明文不可改）
type ProviderKeyUpdateForm struct {
	Name       *string    `json:"name"`
	Priority   *int       `json:"priority"`
	Weight     *int       `json:"weight"`
	Status     *int8      `json:"status"`
	DailyQuota *int       `json:"dailyQuota"`
	RpmLimit   *int       `json:"rpmLimit"`
	ExpiresAt  *time.Time `json:"expiresAt"`
}

// ==================== MCP Server ====================

// McpServerCreateForm 注册外部 MCP Server 表单
type McpServerCreateForm struct {
	Name         string  `json:"name" binding:"required,max=128"`
	Description  *string `json:"description"`
	ProtocolType string  `json:"protocolType"`
	Endpoint     *string `json:"endpoint"`
	AuthType     *string `json:"authType"`
}

// McpServerUpdateForm 更新 Server 表单
type McpServerUpdateForm struct {
	Name         *string `json:"name"`
	Description  *string `json:"description"`
	ProtocolType *string `json:"protocolType"`
	Endpoint     *string `json:"endpoint"`
	AuthType     *string `json:"authType"`
}

// McpServerStatusForm 启停 Server 表单。
// 用指针承接 required：`status=0`（禁用）是合法值，非指针 int8 会被 validator 判为"缺失"而报 A0400
// （python 侧 `status: int = Field(..., ge=0, le=1)` 允许 0）。
type McpServerStatusForm struct {
	Status *int8 `json:"status" binding:"required,oneof=0 1"`
}

// McpNamespaceForm 命名空间配置项
type McpNamespaceForm struct {
	Name      string   `json:"name" binding:"required,max=128"`
	ToolNames []string `json:"toolNames"`
}

// McpCredentialForm 凭据配置表单（加密存储，不回显）
type McpCredentialForm struct {
	ApiKey string            `json:"apiKey"`
	Extra  map[string]string `json:"extra"`
	Clear  bool              `json:"clear"`
}

// McpServerQuery Server 分页查询
type McpServerQuery struct {
	AiPageQuery
	Keyword string `form:"keyword"`
	Status  *int8  `form:"status"`
}

// McpCallQuery 调用审计分页查询
type McpCallQuery struct {
	AiPageQuery
	ServerID *int64 `form:"serverId"`
	ToolName string `form:"toolName"`
}

// ==================== Skill ====================

// SkillCreateForm 创建 Skill 表单
type SkillCreateForm struct {
	Name        string `json:"name" binding:"required,max=128"`
	Description string `json:"description" binding:"required,max=500"`
	Scene       string `json:"scene"`
	Instruction string `json:"instruction" binding:"required"`
}

// SkillUpdateForm 更新 Skill 表单
type SkillUpdateForm struct {
	Name        *string `json:"name"`
	Description *string `json:"description"`
	Scene       *string `json:"scene"`
	Instruction *string `json:"instruction"`
}

// SkillStatusForm 启停 Skill 表单（同 McpServerStatusForm：指针才能让 status=0 通过 required）
type SkillStatusForm struct {
	Status *int8 `json:"status" binding:"required,oneof=0 1"`
}

// SkillShareForm 共享 Skill 至市场表单
type SkillShareForm struct {
	SkillID int64 `json:"skillId" binding:"required"`
}

// SkillQuery Skill 分页查询
type SkillQuery struct {
	AiPageQuery
	Keyword string `form:"keyword"`
	Status  *int8  `form:"status"`
}
