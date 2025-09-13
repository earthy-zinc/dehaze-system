package model

import (
	"time"
)

type BASE_MODEL struct {
	ID        uint      `gorm:"primaryKey;column:id;comment:主键ID" json:"id"`
	CreatedAt time.Time `gorm:"column:create_time" json:"createTime"` // 创建时间
	UpdatedAt time.Time `gorm:"column:update_time" json:"updateTime"` // 更新时间
}
