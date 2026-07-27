package model

// SysDictType 字典类型表
type SysDictType struct {
	BaseModel
	Name   string `gorm:"column:name;type:varchar(50);default:'';comment:类型名称" json:"name"`
	Code   string `gorm:"column:code;type:varchar(50);uniqueIndex:uk_code;default:'';comment:类型编码" json:"code"`
	Status int8   `gorm:"column:status;type:tinyint;default:1;comment:状态(1:启用;0:禁用)" json:"status"`
	Remark string `gorm:"column:remark;type:varchar(255);comment:备注" json:"remark"`
	Deleted int8  `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}
