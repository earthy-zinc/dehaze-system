package model

import "time"

// SysAlgorithmFavorite 算法收藏表（Python 端独有功能，Go 端兼容实现）
type SysAlgorithmFavorite struct {
	ID          int64     `gorm:"primaryKey;autoIncrement;column:id" json:"id"`
	UserID      int64     `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_algorithm;comment:用户ID" json:"userId"`
	AlgorithmID int64     `gorm:"column:algorithm_id;type:bigint;not null;uniqueIndex:uk_user_algorithm;comment:算法ID" json:"algorithmId"`
	CreatedAt   time.Time `gorm:"column:create_time;type:datetime;autoCreateTime;comment:收藏时间" json:"createTime"`
}

func (SysAlgorithmFavorite) TableName() string {
	return "sys_algorithm_favorite"
}
