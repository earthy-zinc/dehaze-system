package model

// SysWpxFile WPX文件表
type SysWpxFile struct {
	BaseModel
	OriginFileID int64  `gorm:"column:origin_file_id;type:bigint;not null;comment:原始文件ID" json:"originFileId"`
	OriginMD5    string `gorm:"column:origin_md5;type:char(32);not null;comment:原始文件MD5" json:"originMd5"`
	OriginPath   string `gorm:"column:origin_path;type:varchar(255);not null;comment:原始文件路径" json:"originPath"`
	NewFileID    int64  `gorm:"column:new_file_id;type:bigint;not null;comment:新文件ID" json:"newFileId"`
	NewPath      string `gorm:"column:new_path;type:varchar(255);not null;comment:新文件路径" json:"newPath"`
	NewMD5       string `gorm:"column:new_md5;type:char(32);not null;comment:新文件MD5" json:"newMd5"`
}
