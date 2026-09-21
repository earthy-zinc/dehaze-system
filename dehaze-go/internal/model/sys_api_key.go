package model

import (
	"encoding/json"
	"time"
)

type SysApiKey struct {
	BaseModel
	UserID     int64      `gorm:"column:user_id;type:bigint;not null" json:"userId"`
	Name       string     `gorm:"column:name;type:varchar(128)" json:"name"`
	KeyPrefix  string     `gorm:"column:key_prefix;type:varchar(16)" json:"keyPrefix"`
	KeyHash    string     `gorm:"column:key_hash;type:varchar(64);uniqueIndex:uk_key_hash" json:"-"`
	Status     int8       `gorm:"column:status;type:tinyint;default:1" json:"status"`
	ExpiresAt  *time.Time `gorm:"column:expires_at;type:datetime" json:"expiresAt"`
	LastUsedAt *time.Time `gorm:"column:last_used_at;type:datetime" json:"lastUsedAt"`
	RevokedAt  *time.Time `gorm:"column:revoked_at;type:datetime;comment:吊销时间(NULL:未吊销)" json:"revokedAt"`
	// Key 级配额（NULL 或 0 = 不限制），列与 python `api_key.py:27-35` 同源
	DailyQuota   *int64 `gorm:"column:daily_quota;type:bigint;comment:日调用配额(NULL或0表示不限制)" json:"dailyQuota"`
	MonthlyQuota *int64 `gorm:"column:monthly_quota;type:bigint;comment:月调用配额(NULL或0表示不限制)" json:"monthlyQuota"`
	RpmLimit     *int   `gorm:"column:rpm_limit;type:int;comment:每分钟请求数上限(NULL或0表示不限制)" json:"rpmLimit"`
	// 模型白名单：JSON 数组，NULL/空数组 = 继承用户可见模型（执行点只在 python 兼容层，见 compatible_governance）
	ModelWhitelist json.RawMessage `gorm:"column:model_whitelist;type:json;comment:模型白名单(NULL或空数组表示继承用户可见模型)" json:"modelWhitelist"`
}

func (SysApiKey) TableName() string {
	return "sys_api_key"
}
