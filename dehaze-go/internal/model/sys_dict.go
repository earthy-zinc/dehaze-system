package model

// SysDict 字典数据表
type SysDict struct {
	BaseModel
	TypeCode  string `gorm:"column:type_code;type:varchar(64);comment:字典类型编码" json:"typeCode"`
	Name      string `gorm:"column:name;type:varchar(50);default:'';comment:字典项名称" json:"name"`
	Value     string `gorm:"column:value;type:varchar(50);default:'';comment:字典项值" json:"value"`
	Sort      int    `gorm:"column:sort;type:int;default:0;comment:排序" json:"sort"`
	Status    int8   `gorm:"column:status;type:tinyint;comment:状态(1:启用;0:禁用)" json:"status"`
	Defaulted int8   `gorm:"column:defaulted;type:tinyint;default:0;comment:是否默认(1:是;0:否)" json:"defaulted"`
	Remark    string `gorm:"column:remark;type:varchar(255);default:'';comment:备注" json:"remark"`
	Deleted   int8   `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}
