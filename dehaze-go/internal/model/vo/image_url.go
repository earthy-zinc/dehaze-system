package vo

// ImageUrlVO 图片URL视图对象
type ImageUrlVO struct {
	// 数据项文件ID
	ID int64 `json:"id"`
	// 所属数据项ID
	ItemID int64 `json:"itemId,omitempty"`
	// 所属数据集ID
	DatasetID int64 `json:"datasetId,omitempty"`
	// 图片类型：clear/hazy/trans/depth/segment
	Type string `json:"type"`
	// 图片访问URL
	URL string `json:"url"`
	// 原始图片URL
	OriginURL string `json:"originUrl,omitempty"`
	// 缩略图URL
	ThumbnailURL string `json:"thumbnailUrl,omitempty"`
	// 图片描述信息
	Description string `json:"description,omitempty"`
	// 图片宽度
	Width int `json:"width,omitempty"`
	// 图片高度
	Height int `json:"height,omitempty"`
	// 场景类型
	SceneType string `json:"sceneType,omitempty"`
	// 雾霾程度
	HazeLevel string `json:"hazeLevel,omitempty"`
	// 文件名
	FileName string `json:"fileName,omitempty"`
	// 文件大小（字节）
	SizeBytes int64 `json:"sizeBytes,omitempty"`
	// 文件格式
	Format string `json:"format,omitempty"`
}
