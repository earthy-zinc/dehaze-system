package model

// SysDataset 数据集表
type SysDataset struct {
	BaseModel
	ParentID    int64  `gorm:"column:parent_id;type:bigint;not null;default:0;comment:父数据集ID" json:"parentId"`
	Type        string `gorm:"column:type;type:varchar(64);not null;default:'';comment:数据集类型" json:"type"`
	Name        string `gorm:"column:name;type:varchar(64);not null;default:'';comment:数据集名称" json:"name"`
	Img         string `gorm:"column:img;type:text;comment:数据集样例图片" json:"img"`
	Description string `gorm:"column:description;type:varchar(2048);default:'';comment:数据集描述" json:"description"`
	Path        string `gorm:"column:path;type:varchar(512);not null;default:'';comment:存储位置" json:"path"`
	Size        string `gorm:"column:size;type:varchar(100);default:'';comment:占用空间大小" json:"size"`
	Status      int8   `gorm:"column:status;type:tinyint;not null;default:1;comment:状态(1:启用；0:禁用)" json:"status"`
	Deleted     int8   `gorm:"column:deleted;type:tinyint;default:0;comment:逻辑删除标识(1:已删除\\;0:未删除)" json:"deleted"`
	CreateBy    int64  `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy    int64  `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
