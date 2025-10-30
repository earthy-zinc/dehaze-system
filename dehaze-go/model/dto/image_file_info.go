package dto

// ImageFileInfo 图片文件信息
type ImageFileInfo struct {
	// 当前图片id
	ID int64 `json:"id"`
	// 所属数据项id
	DatasetItemID int64 `json:"datasetItemId"`
	// 所属文件id
	FileID int64 `json:"fileId"`
	// 图片类型
	Type string `json:"type"`
	// 文件描述
	Description string `json:"description"`
	// 文件URL
	URL string `json:"url"`
}
