package model

// SysEvalLog 评估日志表
type SysEvalLog struct {
	BaseModel
	AlgorithmID int64  `gorm:"column:algorithm_id;type:bigint;not null;comment:算法ID" json:"algorithmId"`
	PredFileID  int64  `gorm:"column:pred_file_id;type:bigint;not null;comment:预测文件ID" json:"predFileId"`
	PredMD5     string `gorm:"column:pred_md5;type:char(32);not null;comment:预测文件MD5" json:"predMd5"`
	PredURL     string `gorm:"column:pred_url;type:text;not null;comment:预测文件URL" json:"predUrl"`
	GtFileID    int64  `gorm:"column:gt_file_id;type:bigint;not null;comment:真实文件ID" json:"gtFileId"`
	GtMD5       string `gorm:"column:gt_md5;type:char(32);not null;comment:真实文件MD5" json:"gtMd5"`
	GtURL       string `gorm:"column:gt_url;type:text;not null;comment:真实文件URL" json:"gtUrl"`
	Time        int    `gorm:"column:time;type:int;comment:耗时(秒)" json:"time"`
	Result      string `gorm:"column:result;type:text;comment:评估结果" json:"result"`
	CreateBy    int64  `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy    int64  `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}
