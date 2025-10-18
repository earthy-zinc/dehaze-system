package model

// SysPredLog 预测日志表
type SysPredLog struct {
	BaseModel
	AlgorithmID  int64  `gorm:"column:algorithm_id;type:bigint;not null;comment:算法ID" json:"algorithmId"`
	OriginFileID int64  `gorm:"column:origin_file_id;type:bigint;not null;comment:原始文件ID" json:"originFileId"`
	OriginMD5    string `gorm:"column:origin_md5;type:char(32);not null;comment:原始文件MD5" json:"originMd5"`
	OriginURL    string `gorm:"column:origin_url;type:text;not null;comment:原始文件URL" json:"originUrl"`
	PredFileID   int64  `gorm:"column:pred_file_id;type:bigint;not null;comment:预测文件ID" json:"predFileId"`
	PredMD5      string `gorm:"column:pred_md5;type:char(32);not null;comment:预测文件MD5" json:"predMd5"`
	PredURL      string `gorm:"column:pred_url;type:text;not null;comment:预测文件URL" json:"predUrl"`
	Time         int    `gorm:"column:time;type:int;comment:耗时(秒)" json:"time"`
	CreateBy     int64  `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy     int64  `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
