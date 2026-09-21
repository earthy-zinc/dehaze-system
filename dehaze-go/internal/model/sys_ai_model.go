package model

import (
	"encoding/json"
	"time"
)

// SysAiModel AI模型配置表
type SysAiModel struct {
	BaseModel
	ProviderID            int64           `gorm:"column:provider_id;type:bigint;not null;index:idx_provider;comment:关联供应商ID" json:"providerId"`
	ModelID               string          `gorm:"column:model_id;type:varchar(64);not null;comment:模型标识" json:"modelId"`
	ModelType             string          `gorm:"column:model_type;type:varchar(16);not null;default:chat;comment:模型类型(chat/embedding/rerank)" json:"modelType"`
	Dimension             *int64          `gorm:"column:dimension;type:bigint;comment:embedding向量维度(创建后不可改)" json:"dimension"`
	DisplayName           string          `gorm:"column:display_name;type:varchar(128);not null;comment:显示名称" json:"displayName"`
	MaxContextTokens      int             `gorm:"column:max_context_tokens;type:int;not null;default:4096;comment:最大上下文Token数" json:"maxContextTokens"`
	MaxOutputTokens       int             `gorm:"column:max_output_tokens;type:int;not null;default:4096;comment:最大输出Token数" json:"maxOutputTokens"`
	SupportsMultimodal    int8            `gorm:"column:supports_multimodal;type:tinyint;not null;default:0;comment:是否支持多模态" json:"supportsMultimodal"`
	SupportsToolCall      int8            `gorm:"column:supports_tool_call;type:tinyint;not null;default:0;comment:是否支持工具调用" json:"supportsToolCall"`
	SupportsStreaming     int8            `gorm:"column:supports_streaming;type:tinyint;not null;default:1;comment:是否支持流式输出" json:"supportsStreaming"`
	SupportsPromptCache   int8            `gorm:"column:supports_prompt_cache;type:tinyint;not null;default:0;comment:是否支持Prompt缓存" json:"supportsPromptCache"`
	SupportsStructuredOut int8            `gorm:"column:supports_structured_output;type:tinyint;not null;default:0;comment:是否支持结构化输出" json:"supportsStructuredOutput"`
	ExtraRequestParams    json.RawMessage `gorm:"column:extra_request_params;type:json;comment:厂商私有请求参数" json:"extraRequestParams"`
	FallbackModelID       *int64          `gorm:"column:fallback_model_id;type:bigint;comment:降级模型ID" json:"fallbackModelId"`
	PromptCachePrefixLen  int             `gorm:"column:prompt_cache_prefix_len;type:int;not null;default:0;comment:Prompt缓存稳定前缀长度" json:"promptCachePrefixLen"`
	ImageTokensPerImage   int             `gorm:"column:image_tokens_per_image;type:int;not null;default:0;comment:多模态单图Token换算上限" json:"imageTokensPerImage"`
	ConcurrencyLimit      *int            `gorm:"column:concurrency_limit;type:int;comment:供应商侧并发上限参考值" json:"concurrencyLimit"`
	Status                int8            `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	LastTestStatus        int8            `gorm:"column:last_test_status;type:tinyint;not null;default:0;comment:最近可用性测试状态" json:"lastTestStatus"`
	LastTestAt            *time.Time      `gorm:"column:last_test_at;type:datetime;comment:最近可用性测试时间" json:"lastTestAt"`
	LastTestError         *string         `gorm:"column:last_test_error;type:varchar(500);comment:最近可用性测试错误信息" json:"lastTestError"`
	VipLevel              int8            `gorm:"column:vip_level;type:tinyint;not null;default:0;comment:最低可用VIP等级" json:"vipLevel"`
	Deleted               int64           `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiModel) TableName() string {
	return "sys_ai_model"
}
