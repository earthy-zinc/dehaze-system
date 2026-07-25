package model

import "time"

// SysPredLog 预测日志表
type SysPredLog struct {
	ID           int64     `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	AlgorithmID  int64     `gorm:"column:algorithm_id;type:bigint;not null;comment:算法id" json:"algorithmId"`
	OriginFileID *int64    `gorm:"column:origin_file_id;type:bigint;comment:原始图像文件id（有雾图像）" json:"originFileId"`
	OriginMD5    string    `gorm:"column:origin_md5;type:char(32);not null;comment:原始图像md5值" json:"originMd5"`
	OriginURL    string    `gorm:"column:origin_url;type:text;not null;comment:原始图像url" json:"originUrl"`
	PredFileID   *int64    `gorm:"column:pred_file_id;type:bigint;comment:预测图像文件id" json:"predFileId"`
	PredMD5      string    `gorm:"column:pred_md5;type:char(32);not null;comment:预测图像md5值" json:"predMd5"`
	PredURL      string    `gorm:"column:pred_url;type:text;not null;comment:预测图像url" json:"predUrl"`
	Time         int       `gorm:"column:time;type:int;default:0;comment:推理时间（秒）" json:"time"`
	Status       string    `gorm:"column:status;type:varchar(20);not null;default:completed;comment:任务状态：processing/completed/failed" json:"status"`
	ErrorMessage *string   `gorm:"column:error_message;type:text;comment:失败错误信息" json:"errorMessage"`
	CreatedAt    time.Time `gorm:"column:create_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
	UpdatedAt    time.Time `gorm:"column:update_time;type:datetime;not null;default:CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy     *int64    `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy     *int64    `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
