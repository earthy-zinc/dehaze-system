package model

type SysPredLog struct {
	BaseModel
	ModelID       string    `gorm:"column:model_id;type:varchar(64);not null;comment:图像模型标识（统一模型规范 slug）" json:"model"`
	ProviderID    *int64    `gorm:"column:provider_id;type:bigint;comment:实际调用供应商id（归因，降级后为备用供应商）" json:"providerId"`
	ActualModel   *string   `gorm:"column:actual_model;type:varchar(64);comment:实际执行模型" json:"actualModel"`
	VendorTaskID  *string   `gorm:"column:vendor_task_id;type:varchar(128);comment:异步厂商任务号（本地同步推理为空）" json:"vendorTaskId"`
	OriginFileID  *int64    `gorm:"column:origin_file_id;type:bigint;comment:原始图像文件id（有雾图像）" json:"originFileId"`
	OriginMD5     string    `gorm:"column:origin_md5;type:char(32);not null;comment:原始图像md5值" json:"originMd5"`
	OriginURL     string    `gorm:"column:origin_url;type:text;not null;comment:原始图像url" json:"originUrl"`
	PredFileID    *int64    `gorm:"column:pred_file_id;type:bigint;comment:预测图像文件id" json:"predFileId"`
	PredMD5       string    `gorm:"column:pred_md5;type:char(32);not null;comment:预测图像md5值" json:"predMd5"`
	PredURL       string    `gorm:"column:pred_url;type:text;not null;comment:预测图像url" json:"predUrl"`
	RecommendedBy *int64    `gorm:"column:recommended_by;type:bigint;comment:推荐来源：推荐记录ID（推荐管理模块，用于追踪推荐采纳率）" json:"recommendedBy"`
	Time          int       `gorm:"column:time;type:int;default:0;comment:推理时间（秒）" json:"time"`
	Status        LogStatus `gorm:"column:status;type:tinyint;not null;comment:任务状态(1:处理中;2:已完成;3:失败;4:已取消)" json:"status"`
	ErrorMessage  *string   `gorm:"column:error_message;type:text;comment:失败错误信息" json:"errorMessage"`
}
