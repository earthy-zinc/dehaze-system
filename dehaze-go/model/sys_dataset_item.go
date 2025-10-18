package model

// SysDatasetItem 数据集项表
type SysDatasetItem struct {
	ID        int64  `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	DatasetID int64  `gorm:"column:dataset_id;type:bigint;not null;comment:所属数据集id" json:"datasetId"`
	Name      string `gorm:"column:name;type:varchar(64);comment:数据项名称" json:"name"`
}
