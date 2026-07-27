package model

// SysFile 文件表
type SysFile struct {
	BaseModel
	Type       *string `gorm:"column:type;type:varchar(100);comment:文件类型" json:"type"`
	URL        *string `gorm:"column:url;type:text;comment:文件url" json:"url"`
	Name       string  `gorm:"column:name;type:varchar(100);not null;comment:文件原始名" json:"name"`
	ObjectName string  `gorm:"column:object_name;type:varchar(100);not null;comment:文件存储名" json:"objectName"`
	Size       string  `gorm:"column:size;type:varchar(100);not null;default:'0';comment:文件大小（格式化显示）" json:"size"`
	SizeBytes  *int64  `gorm:"column:size_bytes;type:bigint;comment:文件大小（原始字节数）" json:"sizeBytes"`
	Path       string  `gorm:"column:path;type:varchar(255);not null;comment:文件路径" json:"path"`
	MD5        string  `gorm:"column:md5;type:char(32);not null;uniqueIndex:uk_md5;comment:文件的MD5值，用于比对文件是否相同" json:"md5"`
	Deleted    int8    `gorm:"column:deleted;type:tinyint;not null;default:0;comment:逻辑删除标识(0:未删除;1:已删除)" json:"deleted"`
}

func (SysFile) TableName() string {
	return "sys_file"
}
