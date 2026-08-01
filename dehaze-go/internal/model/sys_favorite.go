package model

// SysFavorite 统一收藏表
type SysFavorite struct {
	BaseModel
	UserID     int64  `gorm:"column:user_id;type:bigint;not null;uniqueIndex:uk_user_target,priority:1;index:idx_user_type_time,priority:1;comment:用户ID" json:"userId"`
	TargetType string `gorm:"column:target_type;type:varchar(32);not null;uniqueIndex:uk_user_target,priority:2;index:idx_user_type_time,priority:2;comment:收藏对象类型(algorithm/result/dataset/image/preset)" json:"targetType"`
	TargetID   int64  `gorm:"column:target_id;type:bigint;not null;uniqueIndex:uk_user_target,priority:3;comment:收藏对象ID" json:"targetId"`
	IsInvalid  int8   `gorm:"column:is_invalid;type:tinyint;not null;default:0;comment:收藏对象是否已失效(0:正常;1:已失效)" json:"isInvalid"`
	Deleted    int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}

func (SysFavorite) TableName() string {
	return "sys_favorite"
}
