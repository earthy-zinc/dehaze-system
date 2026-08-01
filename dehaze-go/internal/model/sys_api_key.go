package model

import "time"

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
}

func (SysApiKey) TableName() string {
	return "sys_api_key"
}
