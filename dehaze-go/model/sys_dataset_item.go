package model

// SysDatasetItem 数据集项表
type SysDatasetItem struct {
	BaseModel
	DatasetID int64  `gorm:"column:dataset_id;type:bigint;not null;comment:数据集ID" json:"datasetId"`
	Name      string `gorm:"column:name;type:varchar(255);not null;default:'';comment:项名称" json:"name"`
}
