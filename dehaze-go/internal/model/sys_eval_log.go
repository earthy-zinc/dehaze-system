package model

import "time"

// SysEvalLog 评估日志表
type SysEvalLog struct {
	ID          int64     `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	AlgorithmID int64     `gorm:"column:algorithm_id;type:bigint;not null;comment:算法id" json:"algorithmId"`
	PredFileID  *int64    `gorm:"column:pred_file_id;type:bigint;comment:预测图像文件id" json:"predFileId"`
	PredMD5     string    `gorm:"column:pred_md5;type:char(32);not null;comment:预测图像md5值" json:"predMd5"`
	PredURL     string    `gorm:"column:pred_url;type:text;not null;comment:预测图像url" json:"predUrl"`
	GtFileID    *int64    `gorm:"column:gt_file_id;type:bigint;comment:真值图像文件id" json:"gtFileId"`
	GtMD5       string    `gorm:"column:gt_md5;type:char(32);not null;comment:真值图像md5值" json:"gtMd5"`
	GtURL       string    `gorm:"column:gt_url;type:text;not null;comment:真值图像url" json:"gtUrl"`
	Time        int       `gorm:"column:time;type:int;default:0;comment:评估时间（秒）" json:"time"`
	Result      *string   `gorm:"column:result;type:json;comment:预测结果" json:"result"`
	CreatedAt   time.Time `gorm:"column:create_time;type:datetime;not null;default:CURRENT_TIMESTAMP;comment:创建时间" json:"createTime"`
	UpdatedAt   time.Time `gorm:"column:update_time;type:datetime;not null;default:CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP;comment:更新时间" json:"updateTime"`
	CreateBy    *int64    `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy    *int64    `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
