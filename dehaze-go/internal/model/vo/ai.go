package vo

import (
	"encoding/json"
	"time"
)

// ==================== AI 模型 ====================

// AiModelVO 模型视图对象（对齐 python AiModelResult / SDK AiModelVO）
type AiModelVO struct {
	ID                    int64           `json:"id"`
	ProviderID            int64           `json:"providerId"`
	ModelID               string          `json:"modelId"`
	ModelType             string          `json:"modelType"`
	Dimension             *int64          `json:"dimension"`
	DisplayName           string          `json:"displayName"`
	MaxContextTokens      int             `json:"maxContextTokens"`
	MaxOutputTokens       int             `json:"maxOutputTokens"`
	SupportsMultimodal    int8            `json:"supportsMultimodal"`
	SupportsToolCall      int8            `json:"supportsToolCall"`
	SupportsStreaming     int8            `json:"supportsStreaming"`
	SupportsPromptCache   int8            `json:"supportsPromptCache"`
	SupportsStructuredOut int8            `json:"supportsStructuredOutput"`
	ExtraRequestParams    json.RawMessage `json:"extraRequestParams"`
	FallbackModelID       *int64          `json:"fallbackModelId"`
	PromptCachePrefixLen  int             `json:"promptCachePrefixLen"`
	Status                int8            `json:"status"`
	VipLevel              int8            `json:"vipLevel"`
	LastTestStatus        int8            `json:"lastTestStatus"`
	LastTestAt            *time.Time      `json:"lastTestAt"`
	LastTestError         *string         `json:"lastTestError"`
	Calls24h              *int64          `json:"calls24h"`
	SuccessRate24h        *int            `json:"successRate24h"`
	LastCallAt            *time.Time      `json:"lastCallAt"`
	SpeedTier             string          `json:"speedTier"`
	IsFallbackTarget      bool            `json:"isFallbackTarget"`
	CreateTime            time.Time       `json:"createTime"`
}

// ModelPriceDetailVO 价格档位明细视图对象（unitPrice 对齐 python Decimal 的字符串序列化）
type ModelPriceDetailVO struct {
	ID        int64  `json:"id"`
	PriceID   int64  `json:"priceId"`
	TokenType string `json:"tokenType"`
	TimeSlot  string `json:"timeSlot"`
	MinTokens int64  `json:"minTokens"`
	MaxTokens *int64 `json:"maxTokens"`
	UnitPrice string `json:"unitPrice"`
}

// ModelPriceVO 价格版本视图对象
type ModelPriceVO struct {
	ID            int64                `json:"id"`
	ModelID       string               `json:"modelId"`
	ProviderID    int64                `json:"providerId"`
	PriceVersion  int                  `json:"priceVersion"`
	Unit          string               `json:"unit"`
	EffectiveFrom time.Time            `json:"effectiveFrom"`
	EffectiveTo   *time.Time           `json:"effectiveTo"`
	Status        int8                 `json:"status"`
	Details       []ModelPriceDetailVO `json:"details"`
	CreateTime    time.Time            `json:"createTime"`
	UpdateTime    time.Time            `json:"updateTime"`
}

// ==================== AI 供应商 ====================

// UserIdentityForwardVO 用户身份透传配置
type UserIdentityForwardVO struct {
	Enabled bool    `json:"enabled"`
	Field   string  `json:"field"`
	Prefix  *string `json:"prefix"`
	MaxLen  *int    `json:"maxLen"`
}

// ProviderVO 供应商视图对象（health 为 Redis 健康快照运行时聚合，不落库）
type ProviderVO struct {
	ID                  int64           `json:"id"`
	ProviderCode        string          `json:"providerCode"`
	DisplayName         string          `json:"displayName"`
	ApiBaseUrl          string          `json:"apiBaseUrl"`
	ProtocolType        string          `json:"protocolType"`
	AuthType            string          `json:"authType"`
	DefaultHeaders      json.RawMessage `json:"defaultHeaders"`
	SortOrder           int             `json:"sortOrder"`
	HealthCheckEnabled  int8            `json:"healthCheckEnabled"`
	UserIdentityForward json.RawMessage `json:"userIdentityForward"`
	Remark              *string         `json:"remark"`
	Health              *string         `json:"health"`
	Status              int8            `json:"status"`
	CreateTime          time.Time       `json:"createTime"`
	UpdateTime          time.Time       `json:"updateTime"`
}

// ProviderEnabledVO 启用供应商精简视图（登录用户可见，不含内部配置）
type ProviderEnabledVO struct {
	ID           int64   `json:"id"`
	ProviderCode string  `json:"providerCode"`
	DisplayName  string  `json:"displayName"`
	ProtocolType string  `json:"protocolType"`
	Health       *string `json:"health"`
	Status       int8    `json:"status"`
}

// ProviderKeyVO 供应商 API Key 视图对象（不含明文与哈希）
type ProviderKeyVO struct {
	ID         int64      `json:"id"`
	ProviderID int64      `json:"providerId"`
	Name       string     `json:"name"`
	KeyPrefix  *string    `json:"keyPrefix"`
	Status     int8       `json:"status"`
	Priority   int        `json:"priority"`
	Weight     int        `json:"weight"`
	DailyQuota *int       `json:"dailyQuota"`
	RpmLimit   *int       `json:"rpmLimit"`
	ExpiresAt  *time.Time `json:"expiresAt"`
	LastUsedAt *time.Time `json:"lastUsedAt"`
	LastUsedBy *int64     `json:"lastUsedBy"`
	CreateTime time.Time  `json:"createTime"`
	UpdateTime time.Time  `json:"updateTime"`
}

// ==================== MCP Server ====================

// McpServerVO MCP Server 视图对象（credentials 仅写入不回显）
type McpServerVO struct {
	ID                   int64      `json:"id"`
	Name                 string     `json:"name"`
	Description          *string    `json:"description"`
	ProtocolType         string     `json:"protocolType"`
	Endpoint             *string    `json:"endpoint"`
	AuthType             *string    `json:"authType"`
	Status               int8       `json:"status"`
	Health               *string    `json:"health"`
	LastCheckTime        *time.Time `json:"lastCheckTime"`
	ToolCount            int        `json:"toolCount"`
	CredentialConfigured bool       `json:"credentialConfigured"`
	CreateTime           time.Time  `json:"createTime"`
	UpdateTime           time.Time  `json:"updateTime"`
}

// McpNamespaceVO 命名空间视图对象
type McpNamespaceVO struct {
	Name      string   `json:"name"`
	ToolNames []string `json:"toolNames"`
}

// McpCallVO 外部 MCP 调用审计视图对象
type McpCallVO struct {
	ID         int64     `json:"id"`
	UserID     *int64    `json:"userId"`
	ServerID   int64     `json:"serverId"`
	ServerName *string   `json:"serverName"`
	ToolName   string    `json:"toolName"`
	Result     string    `json:"result"`
	LatencyMs  *int      `json:"latencyMs"`
	CreateTime time.Time `json:"createTime"`
}

// ==================== Skill ====================

// SkillFileVO SKILL 目录文件清单项
type SkillFileVO struct {
	Path     string  `json:"path"`
	FileSize int64   `json:"fileSize"`
	FileType *string `json:"fileType"`
}

// SkillVO Skill 视图对象（列表项 instruction 留空，缺失字段省略）
type SkillVO struct {
	ID            int64           `json:"id"`
	Name          string          `json:"name"`
	Description   string          `json:"description"`
	Scene         string          `json:"scene"`
	Instruction   *string         `json:"instruction,omitempty"`
	License       *string         `json:"license,omitempty"`
	Compatibility *string         `json:"compatibility,omitempty"`
	Metadata      json.RawMessage `json:"metadata,omitempty"`
	AllowedTools  *string         `json:"allowedTools,omitempty"`
	Files         []SkillFileVO   `json:"files,omitempty"`
	Status        int8            `json:"status"`
	Source        string          `json:"source"`
	AgentCount    int64           `json:"agentCount"`
	MarketShared  int8            `json:"marketShared"`
	CreateTime    time.Time       `json:"createTime"`
	UpdateTime    time.Time       `json:"updateTime"`
}

// SkillMarketVO Skill 市场目录项
type SkillMarketVO struct {
	SkillID     int64  `json:"skillId"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Scene       string `json:"scene"`
	Enabled     bool   `json:"enabled"`
	AgentCount  int64  `json:"agentCount"`
}

// CompatCallVO AI 兼容端点调用审计视图对象（MongoDB ai_api_call_log）
type CompatCallVO struct {
	ID             string     `json:"id"`
	KeyID          *int64     `json:"keyId"`
	KeyPrefix      string     `json:"keyPrefix"`
	ConversationID *int64     `json:"conversationId"`
	Model          *string    `json:"model"`
	Endpoint       string     `json:"endpoint"`
	Protocol       string     `json:"protocol"`
	IsStream       bool       `json:"isStream"`
	InputTokens    int64      `json:"inputTokens"`
	OutputTokens   int64      `json:"outputTokens"`
	Credits        *float64   `json:"credits"`
	StatusCode     int        `json:"statusCode"`
	DurationMs     int64      `json:"durationMs"`
	ClientIp       string     `json:"clientIp"`
	RequestID      string     `json:"requestId"`
	ErrorMsg       *string    `json:"errorMsg"`
	CreateTime     *time.Time `json:"createTime"`
}
