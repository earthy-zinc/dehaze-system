package model

// SysItemFile 项文件关联表
type SysItemFile struct {
	ID              int64   `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	ItemID          int64   `gorm:"column:item_id;type:bigint;not null;comment:所属数据项id" json:"itemId"`
	FileID          int64   `gorm:"column:file_id;type:bigint;not null;comment:文件id" json:"fileId"`
	ThumbnailFileID *int64  `gorm:"column:thumbnail_file_id;type:bigint;comment:缩略图文件id" json:"thumbnailFileId"`
	Type            string  `gorm:"column:type;type:varchar(64);not null;comment:图片类型（清晰图、雾霾图、分割图等）" json:"type"`
	Description     *string `gorm:"column:description;type:varchar(255);comment:描述" json:"description"`
}
