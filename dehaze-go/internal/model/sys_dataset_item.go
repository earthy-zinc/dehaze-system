package model

// SysDatasetItem 数据集项表
type SysDatasetItem struct {
	BaseModel
	DatasetID int64 `gorm:"column:dataset_id;type:bigint;not null;comment:所属数据集id" json:"datasetId"`
	Name      string `gorm:"column:name;type:varchar(64);comment:数据项名称" json:"name"`
	Deleted   int8  `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}
