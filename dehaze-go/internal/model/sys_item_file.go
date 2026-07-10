package model

// SysItemFile 项文件关联表
type SysItemFile struct {
	ID              int64   `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	ItemID          int64   `gorm:"column:item_id;type:bigint;not null;comment:所属数据项id" json:"itemId"`
	FileID          int64   `gorm:"column:file_id;type:bigint;not null;comment:文件id" json:"fileId"`
	ThumbnailFileID *int64  `gorm:"column:thumbnail_file_id;type:bigint;comment:缩略图文件id" json:"thumbnailFileId"`
	Type            string  `gorm:"column:type;type:varchar(64);not null;comment:图片类型（clear/hazy/depth/segment）" json:"type"`
	HazeLevel       *string `gorm:"column:haze_level;type:varchar(32);comment:雾霾程度（light/medium/heavy），仅type=hazy时有值" json:"hazeLevel"`
	SceneType       *string `gorm:"column:scene_type;type:varchar(64);comment:场景类型" json:"sceneType"`
	Width           *int    `gorm:"column:width;type:int;comment:图片宽度" json:"width"`
	Height          *int    `gorm:"column:height;type:int;comment:图片高度" json:"height"`
	Description     *string `gorm:"column:description;type:varchar(255);comment:描述" json:"description"`
}
