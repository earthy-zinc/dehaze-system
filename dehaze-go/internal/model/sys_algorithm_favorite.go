package model

// SysAlgorithmFavorite 算法收藏表
type SysAlgorithmFavorite struct {
	BaseModel
	UserID      int64 `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_algorithm;comment:用户ID" json:"userId"`
	AlgorithmID int64 `gorm:"column:algorithm_id;type:bigint;not null;uniqueIndex:uk_user_algorithm;comment:算法ID" json:"algorithmId"`
	Deleted     int64 `gorm:"column:deleted;type:bigint;not null;default:0;comment:逻辑删除标识(0:未删除;>0:已删除,值为删除时的行id)" json:"deleted"`
}

func (SysAlgorithmFavorite) TableName() string {
	return "sys_algorithm_favorite"
}
