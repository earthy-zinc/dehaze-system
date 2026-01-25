package model

import (
	"time"
)

// SysDatasetItem 数据集项表
type SysDatasetItem struct {
	ID        int64     `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	DatasetID int64     `gorm:"column:dataset_id;type:bigint;not null;comment:所属数据集id" json:"datasetId"`
	Name      string    `gorm:"column:name;type:varchar(64);comment:数据项名称" json:"name"`
	CreatedAt time.Time `gorm:"column:create_time;type:datetime;comment:创建时间" json:"createdAt"`
	UpdatedAt time.Time `gorm:"column:update_time;type:datetime;comment:更新时间" json:"updatedAt"`
}
