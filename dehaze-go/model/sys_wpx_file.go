package model

// SysWpxFile WPX文件表
type SysWpxFile struct {
	ID           int64  `gorm:"primaryKey;autoIncrement;column:id;comment:id" json:"id"`
	OriginFileID *int64 `gorm:"column:origin_file_id;type:bigint;uniqueIndex:origin_md5;comment:旧文件id" json:"originFileId"`
	OriginMD5    string `gorm:"column:origin_md5;type:char(32);not null;uniqueIndex:origin_md5;comment:旧文件的MD5值" json:"originMd5"`
	OriginPath   string `gorm:"column:origin_path;type:varchar(255);not null;comment:旧文件路径" json:"originPath"`
	NewFileID    *int64 `gorm:"column:new_file_id;type:bigint;uniqueIndex:new_md5;comment:新文件id" json:"newFileId"`
	NewPath      string `gorm:"column:new_path;type:varchar(255);not null;comment:新文件路径" json:"newPath"`
	NewMD5       string `gorm:"column:new_md5;type:char(32);not null;uniqueIndex:new_md5;comment:新文件的MD5值" json:"newMd5"`
}
