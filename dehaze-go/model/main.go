package model

import (
	"time"
)

type BaseModel struct {
	ID        int64     `gorm:"primaryKey;column:id;comment:主键ID" json:"id"`
	CreatedAt time.Time `gorm:"column:create_time;type:datetime" json:"createTime"` // 创建时间
	UpdatedAt time.Time `gorm:"column:update_time;type:datetime" json:"updateTime"` // 更新时间
}
