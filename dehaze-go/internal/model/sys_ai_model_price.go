package model

import "time"

// SysAiModelPrice 模型用户售价版本主表
type SysAiModelPrice struct {
	BaseModel
	ModelID       string     `gorm:"column:model_id;type:varchar(64);not null;comment:模型标识" json:"modelId"`
	ProviderID    int64      `gorm:"column:provider_id;type:bigint;not null;index:idx_provider;comment:供应商ID" json:"providerId"`
	PriceVersion  int        `gorm:"column:price_version;type:int;not null;default:1;comment:价格版本号" json:"priceVersion"`
	Unit          string     `gorm:"column:unit;type:varchar(24);not null;default:credits_per_million;comment:单价单位" json:"unit"`
	EffectiveFrom time.Time  `gorm:"column:effective_from;type:datetime;not null;comment:生效时间" json:"effectiveFrom"`
	EffectiveTo   *time.Time `gorm:"column:effective_to;type:datetime;comment:失效时间(NULL表示当前版本)" json:"effectiveTo"`
	Status        int8       `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:生效;0:停用)" json:"status"`
	Deleted       int64      `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiModelPrice) TableName() string {
	return "sys_ai_model_price"
}

// SysAiModelPriceDetail 用户售价档位明细（token类型 × 上下文分段 × 时段档位）
type SysAiModelPriceDetail struct {
	BaseModel
	PriceID   int64   `gorm:"column:price_id;type:bigint;not null;index:idx_price;comment:价格版本ID" json:"priceId"`
	TokenType string  `gorm:"column:token_type;type:varchar(16);not null;comment:计费类型(input/cached/output)" json:"tokenType"`
	TimeSlot  string  `gorm:"column:time_slot;type:varchar(16);not null;comment:时段档位(peak/idle)" json:"timeSlot"`
	MinTokens int64   `gorm:"column:min_tokens;type:bigint;not null;default:0;comment:上下文分段下界" json:"minTokens"`
	MaxTokens *int64  `gorm:"column:max_tokens;type:bigint;comment:上下文分段上界" json:"maxTokens"`
	UnitPrice float64 `gorm:"column:unit_price;type:decimal(12,4);not null;default:0;comment:单价(积分/百万token)" json:"unitPrice"`
	Deleted   int64   `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识" json:"deleted"`
}

func (SysAiModelPriceDetail) TableName() string {
	return "sys_ai_model_price_detail"
}
