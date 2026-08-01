package model

import "encoding/json"

type SysPreset struct {
	BaseModel
	Name        string          `gorm:"column:name;type:varchar(64);not null;comment:预设名称" json:"name"`
	Type        string          `gorm:"column:type;type:varchar(16);not null;default:custom;comment:预设类型(system:系统预设;custom:用户自定义)" json:"type"`
	AlgorithmID int64           `gorm:"column:algorithm_id;type:bigint;not null;comment:关联算法ID" json:"algorithmId"`
	Params      *json.RawMessage `gorm:"column:params;type:json;comment:参数键值对(JSON)" json:"params"`
	UserID      *int64          `gorm:"column:user_id;type:bigint;comment:所属用户ID(系统预设为空)" json:"userId"`
	IsDefault   int8            `gorm:"column:is_default;type:tinyint;not null;default:0;comment:是否默认预设" json:"isDefault"`
}

func (SysPreset) TableName() string {
	return "sys_preset"
}
