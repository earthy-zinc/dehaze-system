package vo

// ImageItemVO 图片项视图对象
type ImageItemVO struct {
	// DatasetItemId
	ID int64 `json:"id"`
	DatasetID int64 `json:"datasetId"`
	ImgUrl []ImageUrlVO `json:"imgUrl"`
}