package model

// SysDictType 字典类型表
type SysDictType struct {
	BaseModel
	Name   string `gorm:"column:name;type:varchar(50);default:'';comment:类型名称" json:"name"`
	Code   string `gorm:"column:code;type:varchar(50);uniqueIndex:type_code;default:'';comment:类型编码" json:"code"`
	Status int8   `gorm:"column:status;type:tinyint(1);default:0;comment:状态(0:正常;1:禁用)" json:"status"`
	Remark string `gorm:"column:remark;type:varchar(255);comment:备注" json:"remark"`
}
