package model

// SysAlgorithmVersion 算法版本历史表
type SysAlgorithmVersion struct {
	BaseModel
	AlgorithmID int64   `gorm:"column:algorithm_id;type:bigint;not null;comment:关联算法ID" json:"algorithmId"`
	Version     string  `gorm:"column:version;type:varchar(50);not null;comment:版本号" json:"version"`
	ChangeLog   *string `gorm:"column:change_log;type:text;comment:变更日志" json:"changeLog"`
	Status      *int8   `gorm:"column:status;type:tinyint;comment:该版本时的状态" json:"status"`
	ConfigJSON  *string `gorm:"column:config_json;type:json;comment:该版本时的配置JSON" json:"configJson"`
	ModelFileID *int64  `gorm:"column:model_file_id;type:bigint;comment:模型文件ID" json:"modelFileId"`
	IsActive    *int8   `gorm:"column:is_active;type:tinyint;default:0;comment:是否当前活跃版本" json:"isActive"`
	Deleted     int8    `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}

// TableName 指定表名
func (SysAlgorithmVersion) TableName() string {
	return "sys_algorithm_version"
}
