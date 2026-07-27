package model

type SysPredLog struct {
	BaseModel
	AlgorithmID  int64   `gorm:"column:algorithm_id;type:bigint;not null;comment:算法id" json:"algorithmId"`
	OriginFileID *int64  `gorm:"column:origin_file_id;type:bigint;comment:原始图像文件id（有雾图像）" json:"originFileId"`
	OriginMD5    string  `gorm:"column:origin_md5;type:char(32);not null;comment:原始图像md5值" json:"originMd5"`
	OriginURL    string  `gorm:"column:origin_url;type:text;not null;comment:原始图像url" json:"originUrl"`
	PredFileID   *int64  `gorm:"column:pred_file_id;type:bigint;comment:预测图像文件id" json:"predFileId"`
	PredMD5      string  `gorm:"column:pred_md5;type:char(32);not null;comment:预测图像md5值" json:"predMd5"`
	PredURL      string  `gorm:"column:pred_url;type:text;not null;comment:预测图像url" json:"predUrl"`
	Time         int     `gorm:"column:time;type:int;default:0;comment:推理时间（秒）" json:"time"`
	Status       LogStatus `gorm:"column:status;type:tinyint;not null;default:2;comment:任务状态(1:处理中;2:已完成;3:失败)" json:"status"`
	ErrorMessage *string `gorm:"column:error_message;type:text;comment:失败错误信息" json:"errorMessage"`
}
