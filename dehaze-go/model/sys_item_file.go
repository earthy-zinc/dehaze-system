package model

// SysItemFile 项文件关联表
type SysItemFile struct {
	BaseModel
	ItemID          int64  `gorm:"column:item_id;type:bigint;not null;comment:项ID" json:"itemId"`
	FileID          int64  `gorm:"column:file_id;type:bigint;not null;comment:文件ID" json:"fileId"`
	ThumbnailFileID int64  `gorm:"column:thumbnail_file_id;type:bigint;comment:缩略图文件ID" json:"thumbnailFileId"`
	Type            string `gorm:"column:type;type:varchar(64);not null;default:'';comment:文件类型" json:"type"`
	Description     string `gorm:"column:description;type:varchar(255);default:'';comment:描述" json:"description"`
}
