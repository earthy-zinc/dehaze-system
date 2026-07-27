package model

// SysItemFile 项文件关联表
type SysItemFile struct {
	BaseModel
	ItemID          int64   `gorm:"column:item_id;type:bigint;not null;comment:所属数据项id" json:"itemId"`
	FileID          int64   `gorm:"column:file_id;type:bigint;not null;comment:文件id" json:"fileId"`
	ThumbnailFileID *int64  `gorm:"column:thumbnail_file_id;type:bigint;comment:缩略图文件id" json:"thumbnailFileId"`
	Type            string  `gorm:"column:type;type:varchar(64);not null;comment:图片类型：clear(清晰图/GT) / hazy(有雾图) / trans(透射率图) / depth(深度图) / segment(分割图)" json:"type"`
	HazeLevel       *string `gorm:"column:haze_level;type:varchar(32);comment:雾霾程度标识，支持多种规范：light/medium/heavy（人工分级），beta=0.5（β参数），A=0.8,beta=0.2（大气光A+β双参数），空值表示未标注或无雾" json:"hazeLevel"`
	SceneType       *string `gorm:"column:scene_type;type:varchar(64);comment:场景类型" json:"sceneType"`
	Width           *int    `gorm:"column:width;type:int;comment:图片宽度" json:"width"`
	Height          *int    `gorm:"column:height;type:int;comment:图片高度" json:"height"`
	Description     *string `gorm:"column:description;type:varchar(255);comment:描述" json:"description"`
	UsageCount      int64   `gorm:"column:usage_count;type:bigint;default:0;comment:使用次数" json:"usageCount"`
	Deleted         int8    `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}
