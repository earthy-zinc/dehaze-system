package model

// SysInputHistory 图像输入历史记录表
// 对齐 config/sql/schema/sys_input_history.sql 表定义
type SysInputHistory struct {
	BaseModel
	UserID               int64   `gorm:"column:user_id;type:bigint;not null;comment:用户ID" json:"userId"`
	OriginalImageURL     *string `gorm:"column:original_image_url;type:varchar(500);comment:原始图片URL" json:"originalImageUrl"`
	OriginalThumbnailURL *string `gorm:"column:original_thumbnail_url;type:varchar(500);comment:原始缩略图URL" json:"originalThumbnailUrl"`
	ResultImageURL       *string `gorm:"column:result_image_url;type:varchar(500);comment:处理结果图片URL" json:"resultImageUrl"`
	ResultThumbnailURL   *string `gorm:"column:result_thumbnail_url;type:varchar(500);comment:结果缩略图URL" json:"resultThumbnailUrl"`
	ModelID              *string `gorm:"column:model_id;type:varchar(64);comment:图像模型标识（统一模型规范 slug）" json:"modelId"`
	ModelName            *string `gorm:"column:model_name;type:varchar(100);comment:模型名称（冗余）" json:"modelName"`
	ModelParams          *string `gorm:"column:model_params;type:json;comment:模型参数（JSON）" json:"modelParams"`
	ProcessingTime       *int    `gorm:"column:processing_time;type:int;comment:处理耗时（毫秒）" json:"processingTime"`
	Status               *int8   `gorm:"column:status;type:tinyint;default:3;comment:处理状态（1=成功，2=失败，3=处理中），创建时确定，不随处理进度更新" json:"status"`
	InputSource          *string `gorm:"column:input_source;type:varchar(20);comment:图片来源（upload/camera/sample）" json:"inputSource"`
}

// TableName 指定表名
func (SysInputHistory) TableName() string {
	return "sys_input_history"
}
