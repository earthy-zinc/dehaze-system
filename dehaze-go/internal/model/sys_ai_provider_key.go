package model

import "time"

// SysAiProviderKey 供应商 API Key（状态控制，物理删除，无逻辑删除列）
type SysAiProviderKey struct {
	BaseModel
	ProviderID int64      `gorm:"column:provider_id;type:bigint;not null;index:idx_provider;comment:关联供应商ID" json:"providerId"`
	Name       string     `gorm:"column:name;type:varchar(128);not null;comment:Key名称" json:"name"`
	KeyHash    string     `gorm:"column:key_hash;type:char(64);not null;comment:密钥SHA256哈希" json:"keyHash"`
	KeyPrefix  *string    `gorm:"column:key_prefix;type:varchar(16);comment:密钥前缀(展示用)" json:"keyPrefix"`
	KeyCipher  string     `gorm:"column:key_cipher;type:varchar(512);not null;comment:密钥密文(AES-256-CBC base64)" json:"keyCipher"`
	Status     int8       `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	Priority   int        `gorm:"column:priority;type:int;not null;default:0;comment:优先级" json:"priority"`
	Weight     int        `gorm:"column:weight;type:int;not null;default:1;comment:权重" json:"weight"`
	DailyQuota *int       `gorm:"column:daily_quota;type:int;comment:日调用上限" json:"dailyQuota"`
	RpmLimit   *int       `gorm:"column:rpm_limit;type:int;comment:分钟调用频率上限" json:"rpmLimit"`
	ExpiresAt  *time.Time `gorm:"column:expires_at;type:datetime;comment:过期时间" json:"expiresAt"`
	LastUsedAt *time.Time `gorm:"column:last_used_at;type:datetime;comment:最后使用时间" json:"lastUsedAt"`
	LastUsedBy *int64     `gorm:"column:last_used_by;type:bigint;comment:最后使用的用户ID" json:"lastUsedBy"`
}

func (SysAiProviderKey) TableName() string {
	return "sys_ai_provider_key"
}
