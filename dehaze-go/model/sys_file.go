package model

// SysFile 文件表
type SysFile struct {
	BaseModel
	Type       string `gorm:"column:type;type:varchar(100);comment:文件类型" json:"type"`
	URL        string `gorm:"column:url;type:text;comment:文件url" json:"url"`
	Name       string `gorm:"column:name;type:varchar(100);not null;comment:文件原始名" json:"name"`
	ObjectName string `gorm:"column:object_name;type:varchar(100);not null;comment:文件存储名" json:"objectName"`
	Size       string `gorm:"column:size;type:varchar(100);not null;default:'0';comment:文件大小" json:"size"`
	Path       string `gorm:"column:path;type:varchar(255);not null;comment:文件路径" json:"path"`
	MD5        string `gorm:"column:md5;type:char(32);not null;uniqueIndex:md5,md5_key;comment:文件的MD5值，用于比对文件是否相同" json:"md5"`
}
