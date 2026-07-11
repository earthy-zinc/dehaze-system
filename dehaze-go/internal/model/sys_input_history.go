package model

// SysInputHistory 图像输入历史记录表
// 对齐 dehaze-java/config/sql/schema.sql 中 sys_input_history 表定义
type SysInputHistory struct {
	BaseModel
	UserID               int64   `gorm:"column:user_id;type:bigint;not null;comment:用户ID" json:"userId"`
	OriginalImageURL     *string `gorm:"column:original_image_url;type:varchar(500);comment:原始图片URL" json:"originalImageUrl"`
	OriginalThumbnailURL *string `gorm:"column:original_thumbnail_url;type:varchar(500);comment:原始缩略图URL" json:"originalThumbnailUrl"`
	ResultImageURL       *string `gorm:"column:result_image_url;type:varchar(500);comment:处理结果图片URL" json:"resultImageUrl"`
	ResultThumbnailURL   *string `gorm:"column:result_thumbnail_url;type:varchar(500);comment:结果缩略图URL" json:"resultThumbnailUrl"`
	AlgorithmID          *int64  `gorm:"column:algorithm_id;type:bigint;comment:算法ID" json:"algorithmId"`
	AlgorithmName        *string `gorm:"column:algorithm_name;type:varchar(100);comment:算法名称（冗余）" json:"algorithmName"`
	AlgorithmParams      *string `gorm:"column:algorithm_params;type:text;comment:算法参数（JSON）" json:"algorithmParams"`
	ProcessingTime       *int    `gorm:"column:processing_time;type:int;comment:处理耗时（毫秒）" json:"processingTime"`
	Status               *int8   `gorm:"column:status;type:tinyint;default:3;comment:处理状态（1=成功，2=失败，3=处理中）" json:"status"`
	InputSource          *string `gorm:"column:input_source;type:varchar(20);comment:图片来源（upload/camera/sample）" json:"inputSource"`
	IsFavorite           *int8   `gorm:"column:is_favorite;type:tinyint(1);default:0;comment:是否收藏" json:"isFavorite"`
	SyncStatus           *int8   `gorm:"column:sync_status;type:tinyint;default:0;comment:同步状态（0=未同步，1=已同步）" json:"syncStatus"`
	CreateBy             *int64  `gorm:"column:create_by;type:bigint;comment:创建人ID" json:"createBy"`
	UpdateBy             *int64  `gorm:"column:update_by;type:bigint;comment:修改人ID" json:"updateBy"`
}

// TableName 指定表名
func (SysInputHistory) TableName() string {
	return "sys_input_history"
}
