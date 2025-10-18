package vo

// ImageUrlVO 图片URL视图对象
type ImageUrlVO struct {
	// ItemFileId
	ID int64 `json:"id"`
	Type string `json:"type"`
	URL string `json:"url"`
	OriginURL string `json:"originUrl"`
	Description string `json:"description"`
}