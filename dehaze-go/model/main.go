package model

import (
	"time"
)

type BaseModel struct {
	ID        int64     `gorm:"primaryKey;autoIncrement;column:id;comment:主键ID" json:"id"`
	CreatedAt time.Time `gorm:"column:create_time;type:datetime;default:NULL;comment:创建时间" json:"createTime"` // 创建时间
	UpdatedAt time.Time `gorm:"column:update_time;type:datetime;default:NULL;comment:更新时间" json:"updateTime"` // 更新时间
}
