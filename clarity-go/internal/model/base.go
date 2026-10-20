package model

import "time"

// BaseModel 基础模型结构体
// 包含所有表的通用字段，其他模型通过内嵌此结构体继承这些字段
type BaseModel struct {
	ID        int64     `gorm:"primaryKey;autoIncrement;column:id;comment:主键ID" json:"id"`
	CreatedAt time.Time `gorm:"column:create_time;type:datetime;autoCreateTime;comment:创建时间" json:"createTime"` // 创建时间
	UpdatedAt time.Time `gorm:"column:update_time;type:datetime;autoUpdateTime;comment:更新时间" json:"updateTime"` // 更新时间
	CreateBy  int64     `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy  int64     `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
