package bo

// FileBO 文件业务对象
type FileBO struct {
	// 文件名
	Name string `json:"name"`
	// 文件对象名
	ObjectName string `json:"objectName"`
	// 文件扩展名
	Extension string `json:"extension"`
	// 文件MD5值
	MD5 string `json:"md5"`
	// 文件路径
	Path string `json:"path"`
	// 文件大小
	Size int64 `json:"size"`
	// 文件URL
	URL string `json:"url"`
}